# StackFlow用OpenAI互換APIサーバー

## 概要
このサーバーは、LLM、ビジョンモデル、音声合成（TTS）、音声認識（ASR）を含む複数のAIモデルバックエンドをサポートするOpenAI互換APIインターフェースを提供します。

## クイックスタート
1. 依存関係をインストール：
```bash
pip install -r requirements.txt
```

3. サーバーを起動：
```bash
python3 api_server.py  
```

## サポートされているエンドポイント

### チャット補完
- **エンドポイント**: `POST /v1/chat/completions`
- **リクエスト形式**: OpenAI互換チャット補完リクエスト
- **ストリーミング**: サポート

### テキスト補完
- **エンドポイント**: `POST /v1/completions`
- **リクエスト形式**: OpenAI互換補完リクエスト
- **ストリーミング**: サポート

### 音声合成（TTS）
- **エンドポイント**: `POST /v1/audio/speech`
- **パラメータ**:
  - `model`: TTSモデル名
  - `input`: 合成するテキスト
  - `voice`: 音声タイプ
  - `response_format`: 音声フォーマット（mp3、wav等）

### 音声認識（ASR）
- **音声転写**: `POST /v1/audio/transcriptions`
  - 音声を同じ言語のテキストに変換
- **音声翻訳**: `POST /v1/audio/translations`
  - 音声を英語テキストに変換
- **パラメータ**:
  - `file`: 音声ファイル
  - `model`: ASRモデル名
  - `language`（転写のみ）: ソース言語
  - `prompt`: オプションのプロンプト

### モデル一覧
- **エンドポイント**: `GET /v1/models`
- **戻り値**: 利用可能なモデルのリスト

## FAQ

### Q: なぜ「サポートされていないモデル」エラーが出るのですか？
A: モデル名は設定ファイルで設定されたモデルの1つと完全に一致する必要があります。

### Q: ストリーミングレスポンスを有効にするにはどうすればよいですか？
A: チャット/補完エンドポイントのリクエストボディで`"stream": true`を設定してください。

### Q: ASRではどのような音声フォーマットがサポートされていますか？
A: サポートされているフォーマットは、ASRバックエンドの実装によって異なります。

### Q: モデルのメモリ使用量を管理するにはどうすればよいですか？
A: サーバーはLLMモデル用のプールシステムを実装しています。設定で`pool_size`を調整して同時実行インスタンスを制御してください。

## トラブルシューティング
- **ログ**: 詳細なエラーメッセージについてはサーバーログを確認してください
- **モデル初期化**: 必要なバックエンドサービスがすべて実行されていることを確認してください
- **設定**: config.yamlのモデル名とパラメータを再確認してください

## リクエスト例

### チャット補完
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_KEY" \
-d '{
  "model": "qwen2.5-0.5B-p256-ax630c",
  "messages": [{"role": "user", "content": "Hello!"}],
  "temperature": 0.7
}'
```

### 音声合成
```bash
curl -X POST "http://localhost:8001/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "model": "melotts_zh-cn",
    "input": "Hello world!",
    "voice": "alloy"
  }' \
  --output output.mp3
```

## 必要なライブラリ:
- [StackFlow](https://github.com/m5stack/StackFlow)

## ライセンス
- [M5Module-LLM_OpenAI_API- MIT](LICENSE)
