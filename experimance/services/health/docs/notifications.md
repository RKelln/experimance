# Notifications

The health service sends notifications through one or more configurable channels when service
health changes.

## Channels

| Channel | Config key(s) | When active |
|---|---|---|
| Log file | always active | `health_notifications.log` in the standard logs directory |
| ntfy push | `ntfy_topic`, `ntfy_server` | When `ntfy_topic` is set |
| Webhook | `webhook_url`, `webhook_auth_header` | When `webhook_url` is set |

### ntfy Setup

```toml
ntfy_topic  = "my-experimance-alerts"
ntfy_server = "ntfy.sh"   # default; can be a self-hosted server
```

Subscribe on any ntfy client to the topic to receive push notifications.

### Webhook Setup

```toml
webhook_url         = "https://hooks.example.com/endpoint"
webhook_auth_header = "Bearer eyJ..."   # optional
```

The payload is a JSON object with service status fields.

## Notification Triggers

Notifications fire when any of these conditions are met (subject to level filtering and cooldown):

| Event | Default behaviour |
|---|---|
| Service transitions to `ERROR` or `FATAL` | Always notified |
| Service transitions to `UNKNOWN` (missing file) | Notified if `notify_on_unknown = true` (default) |
| Service transitions to `HEALTHY` | Silent by default (`notify_on_healthy = false`) |
| System startup | Notified if `notification_on_startup = true` (default) |
| System shutdown | Notified if `notification_on_shutdown = true` (default) |
| Post-grace-period system summary | Always sent once |

## Level Filtering

`notification_level` sets the minimum severity:

| `notification_level` | Triggers on |
|---|---|
| `"error"` | `ERROR`, `FATAL` |
| `"warning"` | `WARNING`, `ERROR`, `FATAL` (see note below) |
| `"info"` | All statuses |

> Note: `WARNING` does not trigger notifications at level `"warning"` by default. This is an
> intentional design choice to reduce noise. Set `notification_level = "info"` to include warnings.

## Cooldown

To prevent alert fatigue, the health service enforces a minimum time between repeated
notifications for the same service at the same status:

- Default cooldown: `notification_cooldown` (default 300 s / 5 min)
- `ERROR` / `FATAL`: at least `notification_cooldown` seconds between repeats
- `UNKNOWN`: uses `notification_cooldown`
- `HEALTHY`: uses `notification_cooldown`
- Status transitions (e.g. `HEALTHY` → `ERROR`) always notify immediately, regardless of cooldown

## Buffering

When `enable_buffering = true`, notifications are held for `buffer_time` seconds (default 10 s)
before being sent. This collapses rapid status flapping into a single alert.

On service shutdown, buffered notifications are flushed immediately so no alerts are dropped.

## Disabling Notifications

Set `enable_notifications = false` to silence all channels. The log channel is the only channel
that remains active when notifications are disabled (health state is still logged).

## When to Use / When Not to Use

**Use ntfy** when you want phone/desktop push alerts and can subscribe to a topic.

**Use webhook** when you need to integrate with an existing alerting platform (PagerDuty, Slack
incoming webhooks, custom dashboards, etc.).

**Skip both** (log-only) during local development to avoid noisy push notifications on every
service restart.
