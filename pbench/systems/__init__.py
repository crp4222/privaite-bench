"""One module per system under test. Each defines `System` with:

- `setup()`: load models (not timed);
- `detect(text, lang) -> [(start, end, label)]`: the character spans the system
  would mask in that text, from its own detection path;
- `payload(messages, lang) -> messages` (only when `supports_payload`): the
  request after the system's own request-level integration ran, exactly as it
  would reach the model provider; or `{"blocked": reason}` when the product
  refused it by policy, or `{"failed_closed": reason}` when it answered the
  client with its own error and forwarded nothing;
- `info() -> dict`: package versions, model revisions and every non-default knob.

Modules run inside their own virtualenv (see envs/), so they import only their
own library and the stdlib helpers in `_common`.
"""
