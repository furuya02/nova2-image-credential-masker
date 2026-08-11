"""動作確認用のサンプル画像を生成する。

認証情報が写り込んだスクリーンショットを模した画像を作る。
値はすべてダミー（AWS 公式ドキュメントの EXAMPLE 値ベース）で、実在しない。

  python samples/gen_samples.py --output ./images
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

JP_CANDIDATES = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",          # macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Linux
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
]
JP_BOLD_CANDIDATES = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]
MONO_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def font(candidates, size):
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def gen_console_ja(out_dir):
    """AWS コンソール風の日本語画面"""
    W, H = 1280, 720
    img = Image.new("RGB", (W, H), "#ffffff")
    d = ImageDraw.Draw(img)
    jp, jp_b = lambda s: font(JP_CANDIDATES, s), lambda s: font(JP_BOLD_CANDIDATES, s)
    mono = lambda s: font(MONO_CANDIDATES, s)

    d.rectangle([0, 0, W, 56], fill="#232f3e")
    d.text((24, 18), "サービス", font=jp(18), fill="#ffffff")
    d.text((120, 18), "IAM", font=jp_b(18), fill="#ff9900")
    d.text((980, 18), "山田 太郎 @ 123456789012", font=jp(18), fill="#ffffff")

    d.rectangle([0, 56, 220, H], fill="#f2f3f3")
    for i, item in enumerate(["ダッシュボード", "ユーザーグループ", "ユーザー", "ロール", "ポリシー"]):
        d.text((24, 90 + i * 40), item, font=jp(17), fill="#16191f")

    d.text((260, 90), "ユーザー詳細", font=jp_b(28), fill="#16191f")
    d.line([260, 132, W - 40, 132], fill="#d5dbdb", width=1)

    rows = [
        ("ユーザー名", "sin-hirauchi", False),
        ("ARN", "arn:aws:iam::123456789012:user/sin-hirauchi", True),
        ("メールアドレス", "taro.yamada@example.co.jp", False),
        ("作成日", "2026年8月11日 10:24 (UTC+9)", False),
        ("アクセスキーID", "AKIAIOSFODNN7EXAMPLE", True),
    ]
    y = 170
    for name, value, is_mono in rows:
        d.text((260, y), name, font=jp(16), fill="#687078")
        d.text((480, y - 1), value, font=mono(17) if is_mono else jp(17), fill="#16191f")
        y += 46

    d.rectangle([260, y + 20, W - 40, y + 150], outline="#d5dbdb", width=1)
    d.text((280, y + 40), "セキュリティ認証情報", font=jp_b(18), fill="#16191f")
    d.text((280, y + 76), "コンソールパスワード", font=jp(16), fill="#687078")
    d.text((480, y + 75), "P@ssw0rd-Example-2026", font=mono(17), fill="#16191f")

    path = out_dir / "01_console_ja.png"
    img.save(path)
    return path


def gen_terminal_en(out_dir):
    """ターミナル風の英語画面"""
    W, H = 1100, 640
    img = Image.new("RGB", (W, H), "#1e1e1e")
    d = ImageDraw.Draw(img)
    mono = font(MONO_CANDIDATES, 17)

    d.rectangle([0, 0, W, 36], fill="#323233")
    for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        d.ellipse([16 + i * 22, 13, 28 + i * 22, 25], fill=c)
    d.text((W // 2 - 60, 9), "zsh - .env", font=font(MONO_CANDIDATES, 15), fill="#cccccc")

    entries = [
        ("$ cat .env", None),
        ("AWS_ACCOUNT_ID=", "123456789012"),
        ("AWS_ACCESS_KEY_ID=", "AKIAIOSFODNN7EXAMPLE"),
        ("AWS_SECRET_ACCESS_KEY=", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
        ("API_TOKEN=", "sk-proj-8f2Ka9dQzXvB1nLpR7tYuW3e"),
        ("DB_PASSWORD=", "Pa55w0rd!Example"),
        ("MAIL_FROM=", "noreply@example.co.jp"),
    ]
    y = 70
    for key, value in entries:
        color = "#4ec9b0" if value is None else "#9cdcfe"
        d.text((30, y), key, font=mono, fill=color)
        if value:
            d.text((30 + d.textlength(key, font=mono), y), value, font=mono, fill="#ce9178")
        y += 34

    y += 24
    d.text((30, y), "$ aws sts get-caller-identity", font=mono, fill="#4ec9b0")
    y += 34
    for line in ["{", '    "UserId": "AIDAEXAMPLEUSERID",',
                 '    "Account": "123456789012",', "}"]:
        d.text((30, y), line, font=mono, fill="#d4d4d4")
        y += 34

    path = out_dir / "02_terminal_en.png"
    img.save(path)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="./images")
    args = p.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for gen in (gen_console_ja, gen_terminal_en):
        print(gen(out_dir))
