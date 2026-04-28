# Vendored: DeepSeek-V4 Encoding

`encoding_dsv4.py` is a **verbatim** copy of the upstream encoder/parser shipped
with the DeepSeek-V4 model release.

## Source

- Repo: [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- Path: `encoding/encoding_dsv4.py`
- URL: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/encoding/encoding_dsv4.py>

## Pinned revision

| Field | Value |
| --- | --- |
| Vendored at | 2026-04-28 |
| SHA-256 | `bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0` |
| Lines | 744 |

## License

Upstream is MIT-licensed (same as solar-host). The original copyright/license
applies; see <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/LICENSE>.

## Updating

1. Re-download:

   ```bash
   curl -fsSL https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/encoding/encoding_dsv4.py \
     -o solar_host/servers/deepseek_v4/encoding_dsv4.py
   ```

2. Update the SHA-256 above:

   ```bash
   sha256sum solar_host/servers/deepseek_v4/encoding_dsv4.py
   ```

3. Re-run the unit tests (DeepSeek-V4 chat completion path) before committing.
