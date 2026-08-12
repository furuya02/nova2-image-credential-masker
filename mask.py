"""画像に写り込んだ認証情報を Amazon Nova 2 Lite で検出し、マスクする。

処理フロー（画像 1 枚あたり）:
  1. スクリーニング  認証情報があるかだけを判定。無ければコピーのみで終了（コスト削減）
  2. 検出            認証情報のバウンディングボックスを [0,1000] 正規化座標で取得
  3. 12 桁スキャン   テキストを文字起こしして、正規表現で 12 桁の数字を含む行を拾う
                     （判定をモデルに委ねないため確実。該当行はまるごとマスクする）
  4. マスク          Pillow で塗り潰し（座標を実寸へ変換し、パディングを付与）
  5. 再検証          マスク済み画像を再度 Nova に渡し、残存があれば _review/ へ振り分け

使い方:
  python mask.py --input ./images --output ./masked
"""

import argparse
import json
import re
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
# 実測の不足量（左 1.24）を確実に覆いつつ、隠す範囲が必要以上に広がらない値を既定とする。
DEFAULT_PAD_X = 1.5
DEFAULT_PAD_Y = 0.4

# 塗り潰しの方式。既定は black（該当領域を単色で置き換えるため確実）。
# blur はガウスぼかし。見た目を優先したい場合の選択肢であり、
# 認証情報を隠す用途では black を使うこと。
DEFAULT_STYLE = "black"
BLUR_RADIUS_RATIO = 0.5   # ぼかし半径。検出枠の高さに対する比率

# 応答の上限トークン数。検出・再検証は座標付き JSON を返すため、
# 認証情報が多く写り込んだ画像ではそれなりの長さになる。
# 上限に達すると JSON が途中で終わり解析できないので、余裕を持たせている。
DEFAULT_MAX_TOKENS = 4000

# 文字起こしの実行回数。同じ画像でも拾える行が毎回変わるため、
# 複数回まわして重ね合わせる。取りこぼしを減らすことを優先した既定値。
DEFAULT_OCR_PASSES = 3

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

CATEGORIES = """- aws_account_id: a run of exactly 12 consecutive digits. It may stand alone, or
  sit inside a longer identifier such as arn:aws:iam::123456789012:user/foo -
  report those as well. Cover ONLY the 12 digits: the characters before and after
  them must stay readable. "text" must be the 12 digits alone.
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

# 12 桁の数字を確実に拾うための OCR パス。
# 「12 桁かどうか」の判定はモデルに任せず、文字起こしした文字列に対して
# コード側の正規表現で行う。モデルは長い識別子を 1 つの塊として扱い、
# その中の数字の並びを取り出せないことがあるため。
OCR_PROMPT = """Transcribe the text in this screenshot, line by line.

Rules:
- Copy each line exactly as it appears, including punctuation and long identifiers.
- Do not summarise, translate, or omit anything.
- Give each line a bounding box in the normalized [0, 1000] coordinate space,
  as [x1, y1, x2, y2] where (x1, y1) is the top-left corner.

Respond with JSON only, no markdown fence. Every element must be an object that
has BOTH "text" and "bbox" - never a bare string:
{"lines": [
  {"text": "first line as it appears", "bbox": [100, 200, 400, 230]},
  {"text": "second line as it appears", "bbox": [100, 240, 350, 270]}
]}"""

# 文字起こしが座標なしで返ってきたときに、位置だけを聞き直すためのプロンプト。
LOCATE_PROMPT = """Find where each of the following strings appears in this
screenshot, and give its bounding box.

Strings to locate:
{targets}

Rules:
- Treat each string as an exact target. Locate the whole string.
- Give the bounding box in the normalized [0, 1000] coordinate space,
  as [x1, y1, x2, y2] where (x1, y1) is the top-left corner.
- If a string appears more than once, report each occurrence.

Respond with JSON only, no markdown fence. Every element must be an object:
{{"found": [{{"text": "...", "bbox": [x1, y1, x2, y2]}}]}}"""

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


def converse(client, model_id, img_bytes, fmt, prompt, usage, max_tokens=DEFAULT_MAX_TOKENS):
    """Nova を呼び出して JSON を受け取る。解析できなかった場合は None を返す。"""
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
    # 上限に達した応答は JSON が途中で終わっているため使えない。
    # 検出項目が多い画像で起きやすく、--max-tokens で引き上げられる。
    if resp.get("stopReason") == "max_tokens":
        return None
    return parse_json(resp["output"]["message"]["content"][0]["text"])


def parse_json(text):
    """応答から JSON を取り出す。解析できなければ None を返す。

    指定した形（オブジェクト）で返らないことがある。とくに該当なしの場合、
    {"remaining": []} ではなく [] だけを返してくることがあるため、
    オブジェクトと配列の両方を受け取れるようにしている。
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = t.find(opener), t.rfind(closer)
        if 0 <= start < end:
            candidates.append((start, t[start:end + 1]))
    # 先に現れる方が外側。[{...}] で中のオブジェクトだけを拾わないようにする
    for _, chunk in sorted(candidates):
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue
    return None


def items_of(data, key):
    """{"key": [...]} でも [...] でも、リストとして受け取れるようにする。"""
    if isinstance(data, dict):
        return data.get(key) or []
    if isinstance(data, list):
        return data
    return []


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
        if not isinstance(r, dict):
            continue
        bbox = r.get("bbox")
        if not valid_bbox(bbox):
            real.append(r)
            continue
        cx = (bbox[0] + bbox[2]) / 2 / 1000 * w
        cy = (bbox[1] + bbox[3]) / 2 / 1000 * h
        if not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes):
            real.append(r)
    return real


# 12 桁の数字。AWS コンソールでは 1234-5678-9012 のように区切って表示されることが
# あるため、ハイフンや空白で区切られた形も拾う。
DIGIT_RUN = re.compile(r"\d{12}|\d{4}[\s-]\d{4}[\s-]\d{4}")


def valid_bbox(bbox):
    """座標として使える形か確かめる。モデルの応答は形が崩れることがある。"""
    return (isinstance(bbox, (list, tuple)) and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox))


def scan_digit_runs(client, args, img_bytes, fmt, usage):
    """12 桁の数字を含む行を、文字起こしと正規表現で拾う。

    検出プロンプトだけでは、長い識別子に埋め込まれた数字を取りこぼす。
    ここでは判定をモデルに委ねず、文字起こししたテキストに正規表現をかけ、
    該当した行はまるごとマスク対象にする。範囲は広くなるが確実に消せる。

    文字起こし自体は実行のたびに揺らぎ、1 回では拾い切れないことがある。
    そのため既定で複数回実行し、結果を重ね合わせる（--ocr-passes）。
    """
    hits = []
    for _ in range(max(1, args.ocr_passes)):
        hits += scan_once(client, args, img_bytes, fmt, usage)
    return dedupe_hits(hits)


def dedupe_hits(hits):
    """同じ箇所を指す結果をまとめる。座標が近いものは同一とみなす。"""
    kept = []
    for h in hits:
        x1, y1, x2, y2 = h["bbox"]
        if any(abs(x1 - k["bbox"][0]) < 20 and abs(y1 - k["bbox"][1]) < 20
               and abs(x2 - k["bbox"][2]) < 20 and abs(y2 - k["bbox"][3]) < 20
               for k in kept):
            continue
        kept.append(h)
    return kept


def scan_once(client, args, img_bytes, fmt, usage):
    data = converse(client, args.model_id, img_bytes, fmt, OCR_PROMPT,
                    usage, max_tokens=args.max_tokens)
    lines = items_of(data, "lines")

    # 座標付きで返ってきた場合はそのまま使う
    hits = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        text, bbox = line.get("text", ""), line.get("bbox")
        if valid_bbox(bbox) and DIGIT_RUN.search(str(text)):
            hits.append({"category": "digits12", "text": text, "bbox": bbox})
    if hits:
        return hits

    # 座標なし（文字列だけ）で返ることがある。その場合は該当行の位置を聞き直す。
    targets = []
    for line in lines:
        text = line.get("text", "") if isinstance(line, dict) else line
        if isinstance(text, str) and DIGIT_RUN.search(text):
            targets.append(text)
    if not targets:
        return []
    return locate_texts(client, args, img_bytes, fmt, targets, usage)


def locate_texts(client, args, img_bytes, fmt, targets, usage):
    """指定した文字列が画像のどこにあるかを聞き、座標を得る。"""
    prompt = LOCATE_PROMPT.format(
        targets="\n".join(f"- {s}" for s in targets))
    data = converse(client, args.model_id, img_bytes, fmt, prompt,
                    usage, max_tokens=args.max_tokens)
    hits = []
    for item in items_of(data, "found"):
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if valid_bbox(bbox):
            hits.append({"category": "digits12",
                         "text": item.get("text", ""), "bbox": bbox})
    return hits


def merge_findings(findings, extra):
    """重なりの大きい枠を捨てて統合する。同じ箇所を二重に塗らないため。"""
    findings = [f for f in findings
                if isinstance(f, dict) and valid_bbox(f.get("bbox"))]
    merged = list(findings)
    for e in extra:
        ex1, ey1, ex2, ey2 = e["bbox"]
        for f in findings:
            fx1, fy1, fx2, fy2 = f["bbox"]
            # 既存の枠が新しい枠にすっぽり入るなら、行全体で塗るので不要
            if fx1 >= ex1 and fy1 >= ey1 and fx2 <= ex2 and fy2 <= ey2:
                merged = [m for m in merged if m is not f]
        merged.append(e)
    return merged


def to_review(path, review_dir, reason, findings=0, out_path=None):
    """判断できなかった画像を要確認へ回す。応答を解析できない場合は安全側に倒す。"""
    review_dir.mkdir(parents=True, exist_ok=True)
    if out_path and out_path.exists():
        shutil.move(str(out_path), review_dir / path.name)
    else:
        shutil.copy2(path, review_dir / path.name)
    return "review", findings, [{"category": "-", "text": reason}], 0


def process(path, out_dir, review_dir, client, args, usage):
    img_bytes = path.read_bytes()
    fmt = image_format(path)

    # 12 桁スキャンはスクリーニングより先に、無条件で実行する。
    # スクリーニングは「認証情報らしさ」で判断するため、識別子やリソース名に
    # 埋もれた数字しか写っていない画面を「対象なし」と判定してしまう。
    # 数字の判定は正規表現で完結するので、モデルの判断を待つ必要がない。
    digit_hits = [] if args.no_digit_scan else scan_digit_runs(
        client, args, img_bytes, fmt, usage)

    if not args.no_screen and not digit_hits:
        screened = converse(client, args.model_id, img_bytes, fmt,
                            SCREEN_PROMPT, usage, max_tokens=100)
        # 判定できなかったときは検出へ進める（対象なしと決めつけない）
        if isinstance(screened, dict) and not screened.get("has_credentials"):
            shutil.copy2(path, out_dir / path.name)
            return "clean", 0, [], 0

    detected = converse(client, args.model_id, img_bytes, fmt, DETECT_PROMPT,
                        usage, max_tokens=args.max_tokens)
    if detected is None:
        return to_review(path, review_dir, "検出結果を解析できませんでした"
                                           "（--max-tokens の引き上げをお試しください）")

    findings = [f for f in items_of(detected, "findings")
                if isinstance(f, dict) and valid_bbox(f.get("bbox"))]

    findings = merge_findings(findings, digit_hits)
    n_digits = len(digit_hits)

    if not findings:
        shutil.copy2(path, out_dir / path.name)
        return "clean", 0, [], n_digits

    img = Image.open(path).convert("RGB")
    boxes = apply_mask(img, findings, args.padding_x, args.padding_y, args.style)
    out_path = out_dir / path.name
    img.save(out_path)

    if args.no_verify:
        return "masked", len(findings), [], n_digits

    verified = converse(client, args.model_id, out_path.read_bytes(),
                        image_format(out_path), VERIFY_PROMPT, usage,
                        max_tokens=args.max_tokens)
    if verified is None:
        return to_review(path, review_dir, "再検証の結果を解析できませんでした"
                                           "（--max-tokens の引き上げをお試しください）",
                         len(findings), out_path)

    remaining = drop_hallucinations(items_of(verified, "remaining"), boxes, img.size)
    if remaining:
        review_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out_path), review_dir / path.name)
        return "review", len(findings), remaining, n_digits
    return "masked", len(findings), [], n_digits


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
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"検出・再検証の応答上限トークン数（既定 {DEFAULT_MAX_TOKENS}）")
    p.add_argument("--style", choices=["black", "blur"], default=DEFAULT_STYLE,
                   help="塗り潰しの方式（既定 black）。認証情報には black を推奨")
    p.add_argument("--ocr-passes", type=int, default=DEFAULT_OCR_PASSES,
                   help=f"文字起こしを何回行うか（既定 {DEFAULT_OCR_PASSES}）。"
                        "多いほど取りこぼしは減るが呼び出しが増える")
    p.add_argument("--no-digit-scan", action="store_true",
                   help="12 桁の数字を文字起こしから探すパスを省略する（呼び出しが 1 回減る）")
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
        digits = 0
        try:
            status, n, remaining, digits = process(path, out_dir, review_dir,
                                                   client, args, usage)
        except Exception as e:
            # 1 枚の失敗で全体を止めない。処理できなかった画像は要確認へ回す
            status, n, remaining, digits = to_review(
                path, review_dir, f"処理中にエラーが発生しました（{type(e).__name__}: {e}）")
        tally[status] += 1
        mark = {"clean": "-", "masked": "OK", "review": "!!"}[status]
        print(f"  [{mark}] {path.name}  findings={n}")
        if digits:
            print(f"        12 桁スキャン: {digits} 件")
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
