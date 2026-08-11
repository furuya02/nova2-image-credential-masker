"""画像に写り込んだ認証情報を Amazon Nova 2 Lite で検出し、マスクする。

処理フロー（画像 1 枚あたり）:
  1. スクリーニング  認証情報があるかだけを判定。無ければコピーのみで終了（コスト削減）
  2. 検出            認証情報のバウンディングボックスを [0,1000] 正規化座標で取得
  3. マスク          Pillow で塗り潰し（座標を実寸へ変換し、パディングを付与）
  4. 再検証          マスク済み画像を再度 Nova に渡し、残存があれば _review/ へ振り分け

使い方:
  python mask.py --input ./images --output ./masked
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import boto3
from PIL import Image, ImageDraw, ImageFilter

# Nova 2 Lite は ON_DEMAND 非対応のため推論プロファイル経由で呼び出す。
# jp. プレフィックスは ap-northeast-1 / ap-northeast-3 にのみルーティングされるため、
# 機密画像の処理を日本国内リージョンに閉じられる。
DEFAULT_MODEL_ID = "jp.amazon.nova-2-lite-v1:0"
DEFAULT_REGION = "ap-northeast-1"

# 東京リージョンの Nova 2 Lite 単価（USD / 1M トークン）。
# aws pricing get-products --service-code AmazonBedrock で取得した実値。
PRICE_IN_PER_1M = 0.396
PRICE_OUT_PER_1M = 3.311

# 検出枠は正解に対してずれるため、パディングを付けて黒塗りする。
# パディングは「検出枠の高さ」を基準にした比率で指定する。画像の解像度ではなく
# 文字サイズに追従するため、高解像度の画像でも比率を変えずに済む。
#
# 手元の実測（サンプル 2 枚 / 12 箇所）での不足量は以下だった。
#   横方向: 最大 1.24 x 高さ（左端が 1〜2 文字分はみ出すケースがある）
#   縦方向: 最大 0.01 x 高さ（ほぼ常に覆えている）
# これに余裕を加えた値を既定とする。
DEFAULT_PAD_X = 1.5
DEFAULT_PAD_Y = 0.3

# 塗り潰しの方式。既定は black（該当領域を単色で置き換えるため確実）。
# blur はガウスぼかし。見た目を優先したい場合の選択肢であり、
# 認証情報を隠す用途では black を使うこと。
DEFAULT_STYLE = "black"
BLUR_RADIUS_RATIO = 0.5   # ぼかし半径。検出枠の高さに対する比率

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

CATEGORIES = """- aws_account_id: a 12-digit AWS account number. It may stand alone, or appear
  inside an ARN such as arn:aws:iam::123456789012:user/foo. When it appears inside
  an ARN, the box must cover ONLY the 12 consecutive digits - not the whole ARN.
  The surrounding parts of the ARN must remain readable, and "text" must be the
  12 digits alone.
- aws_access_key_id: an access key ID such as AKIA...
- aws_secret_access_key: a long secret key string
- api_token: an API key, bearer token, or session token
- password: a password value
- email: an email address
- phone: a phone number"""

SCREEN_PROMPT = f"""Look at this screenshot and decide whether it contains any
credential or sensitive identifier rendered as visible text.

Target categories:
{CATEGORIES}

The screenshot may be in Japanese or English.
Respond with JSON only: {{"has_credentials": true}} or {{"has_credentials": false}}"""

DETECT_PROMPT = f"""You are a security screening tool. Find every credential or
sensitive identifier that is visibly rendered as text in this screenshot.

Target categories:
{CATEGORIES}

Rules:
- Report only the VALUE, never the label or field name next to it.
- The screenshot may be in Japanese or English.
- Return a bounding box for each finding using the normalized [0, 1000] coordinate
  space, as [x1, y1, x2, y2] where (x1, y1) is the top-left corner.
- The box must tightly enclose the rendered text of the value.

Respond with JSON only, no markdown fence, in this exact shape:
{{"findings": [{{"category": "...", "text": "...", "bbox": [x1, y1, x2, y2]}}]}}
If there is nothing to report, return {{"findings": []}}."""

VERIFY_PROMPT = f"""This screenshot has already been redacted: sensitive values were
hidden behind solid black boxes or blurred areas. Find any value the redaction MISSED.

Target categories:
{CATEGORIES}

Rules:
- Report a value only if you can actually read its characters in the image.
- Give the bounding box of each finding in the normalized [0, 1000] coordinate
  space, as [x1, y1, x2, y2].
- If every sensitive value is covered, return an empty list.

Respond with JSON only, no markdown fence:
{{"remaining": [{{"category": "...", "text": "...", "bbox": [x1, y1, x2, y2]}}]}}"""


class Usage:
    """トークン使用量を積算してコストを概算する。"""

    def __init__(self):
        self.input = 0
        self.output = 0
        self.calls = 0

    def add(self, usage):
        self.input += usage["inputTokens"]
        self.output += usage["outputTokens"]
        self.calls += 1

    def usd(self):
        return (self.input * PRICE_IN_PER_1M + self.output * PRICE_OUT_PER_1M) / 1_000_000


def converse(client, model_id, img_bytes, fmt, prompt, usage, max_tokens=2000):
    resp = client.converse(
        modelId=model_id,
        messages=[{
            "role": "user",
            "content": [
                {"image": {"format": fmt, "source": {"bytes": img_bytes}}},
                {"text": prompt},
            ],
        }],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )
    usage.add(resp["usage"])
    return parse_json(resp["output"]["message"]["content"][0]["text"])


def parse_json(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    return json.loads(t[start:end + 1])


def image_format(path):
    ext = path.suffix.lower()
    return "jpeg" if ext in (".jpg", ".jpeg") else ext.lstrip(".")


def apply_mask(img, findings, pad_x_ratio, pad_y_ratio, style=DEFAULT_STYLE):
    """[0,1000] 正規化座標を実寸へ変換し、パディングを付けて塗り潰す。"""
    w, h = img.size
    d = ImageDraw.Draw(img)
    boxes = []
    for f in findings:
        x1, y1, x2, y2 = f["bbox"]
        px1, py1 = x1 / 1000 * w, y1 / 1000 * h
        px2, py2 = x2 / 1000 * w, y2 / 1000 * h
        bh = py2 - py1                      # 検出枠の高さ = 文字サイズの目安
        pad_x, pad_y = bh * pad_x_ratio, bh * pad_y_ratio
        box = [max(0, px1 - pad_x), max(0, py1 - pad_y),
               min(w, px2 + pad_x), min(h, py2 + pad_y)]
        if style == "blur":
            # 見た目を優先したい場合の選択肢。認証情報には black を推奨する。
            region_box = tuple(int(v) for v in box)
            region = img.crop(region_box)
            img.paste(region.filter(ImageFilter.GaussianBlur(bh * BLUR_RADIUS_RATIO)),
                      region_box)
        else:
            d.rectangle(box, fill="black")
        boxes.append(box)
    return boxes


def drop_hallucinations(remaining, boxes, size):
    """黒塗り済み領域を指す残存報告を落とす。

    再検証にマスク後の画像を渡すと、Nova は黒塗りの下にあったはずの値を文脈から
    推測して「まだ読める」と報告してくることがある。プロンプトで禁止しても消えない
    ため、報告された座標が黒塗り矩形の内側なら機械的にハルシネーションとみなす。
    マスクがずれて本当に読めてしまっている場合は矩形の外側に出るため検知できる。
    """
    w, h = size
    real = []
    for r in remaining:
        bbox = r.get("bbox")
        if not bbox:
            real.append(r)
            continue
        cx = (bbox[0] + bbox[2]) / 2 / 1000 * w
        cy = (bbox[1] + bbox[3]) / 2 / 1000 * h
        if not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes):
            real.append(r)
    return real


def process(path, out_dir, review_dir, client, args, usage):
    img_bytes = path.read_bytes()
    fmt = image_format(path)

    if not args.no_screen:
        screened = converse(client, args.model_id, img_bytes, fmt,
                            SCREEN_PROMPT, usage, max_tokens=100)
        if not screened.get("has_credentials"):
            shutil.copy2(path, out_dir / path.name)
            return "clean", 0, []

    detected = converse(client, args.model_id, img_bytes, fmt, DETECT_PROMPT, usage)
    findings = detected.get("findings", [])
    if not findings:
        shutil.copy2(path, out_dir / path.name)
        return "clean", 0, []

    img = Image.open(path).convert("RGB")
    boxes = apply_mask(img, findings, args.padding_x, args.padding_y, args.style)
    out_path = out_dir / path.name
    img.save(out_path)

    if args.no_verify:
        return "masked", len(findings), []

    verified = converse(client, args.model_id, out_path.read_bytes(),
                        image_format(out_path), VERIFY_PROMPT, usage)
    remaining = drop_hallucinations(verified.get("remaining", []), boxes, img.size)
    if remaining:
        review_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out_path), review_dir / path.name)
        return "review", len(findings), remaining
    return "masked", len(findings), []


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True, help="入力フォルダ")
    p.add_argument("--output", required=True, help="出力フォルダ")
    p.add_argument("--padding-x", type=float, default=DEFAULT_PAD_X,
                   help=f"横方向のパディング（検出枠の高さに対する倍率, 既定 {DEFAULT_PAD_X}）")
    p.add_argument("--padding-y", type=float, default=DEFAULT_PAD_Y,
                   help=f"縦方向のパディング（検出枠の高さに対する倍率, 既定 {DEFAULT_PAD_Y}）")
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--style", choices=["black", "blur"], default=DEFAULT_STYLE,
                   help="塗り潰しの方式（既定 black）。認証情報には black を推奨")
    p.add_argument("--no-screen", action="store_true",
                   help="スクリーニングを省略して全画像を検出にかける")
    p.add_argument("--no-verify", action="store_true", help="マスク後の再検証を省略する")
    args = p.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    images = sorted(f for f in in_dir.iterdir() if f.suffix.lower() in EXTS)
    if not images:
        print(f"画像が見つかりません: {in_dir}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir = out_dir / "_review"
    client = boto3.client("bedrock-runtime", region_name=args.region)
    usage = Usage()
    tally = {"clean": 0, "masked": 0, "review": 0}

    print(f"model={args.model_id} region={args.region} images={len(images)}\n")
    for path in images:
        status, n, remaining = process(path, out_dir, review_dir, client, args, usage)
        tally[status] += 1
        mark = {"clean": "-", "masked": "OK", "review": "!!"}[status]
        print(f"  [{mark}] {path.name}  findings={n}")
        for r in remaining:
            print(f"        残存の疑い: {r.get('category')} {str(r.get('text'))[:40]!r}")

    print(f"\nマスク済み: {tally['masked']}  対象なし: {tally['clean']}  "
          f"要確認: {tally['review']}")
    if tally["review"]:
        print(f"  要確認の画像は {review_dir} を目視してください")
    print(f"呼び出し {usage.calls} 回 / 入力 {usage.input} tok / 出力 {usage.output} tok")
    print(f"概算コスト ${usage.usd():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
