# --- HTML TEMPLATE FOR DASHBOARD ---
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>GenAI Peers Trust Monitoring</title>
    <meta http-equiv="refresh" content="5">
    <meta charset="UTF-8">
    <style>
    :root{
      --bg: #0b0f14;
      --panel: rgba(18, 24, 33, 0.92);
      --panel2: rgba(24, 31, 42, 0.92);
      --border: rgba(255,255,255,0.08);
      --text: rgba(240,250,255,0.92);
      --muted: rgba(240,250,255,0.65);

      --green: #35e07a;
      --yellow: #ffd84d;
      --red: #ff4d4d;
      --cyan: #00ccff;

      --radius: 14px;
    }

    * { box-sizing: border-box; }

    body{
      margin: 0;
      padding: 22px;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background:
        radial-gradient(1200px 600px at 20% 0%, rgba(0, 204, 255, 0.12), transparent 55%),
        radial-gradient(900px 500px at 90% 10%, rgba(53, 224, 122, 0.08), transparent 50%),
        var(--bg);
      color: var(--text);
    }

    .container{
      max-width: 1200px;
      margin: 0 auto;
    }

    /* Header card */
    .topbar{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px;
      border-radius: var(--radius);
      background: linear-gradient(180deg, rgba(0,43,54,0.95), rgba(0,43,54,0.75));
      border: 1px solid rgba(0, 204, 255, 0.18);
      box-shadow: 0 10px 28px rgba(0,0,0,0.30);
      margin-bottom: 18px;


    }

    .lab-title{
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.2px;
      line-height: 1.15;
    }

    .lab-subtitle{
      margin-top: 6px;
      font-size: 13.5px;
      color: rgba(240,250,255,0.75);
    }

    .page-title{
      margin: 0;
      font-size: 20px;
      font-weight: 750;
      letter-spacing: 0.2px;
      color: rgba(240,250,255,0.95);
    }

    .page-sub{
      margin-top: 6px;
      font-size: 13px;
      color: var(--muted);
    }

    /* Panels */
    .panel{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: 0 12px 30px rgba(0,0,0,0.30);
      overflow: hidden;
    }

    .panel-header{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap: 12px;
      padding: 14px 16px;
      background: rgba(255,255,255,0.03);
      border-bottom: 1px solid var(--border);
    }

    .panel-header h3{
      margin: 0;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.3px;
      color: rgba(240,250,255,0.85);
      text-transform: uppercase;
    }

    .hint{
      font-size: 12.5px;
      color: var(--muted);
      margin: 0;
      white-space: nowrap;
    }

    /* Table */
    table{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13.5px;
    }

    thead th{
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(10, 14, 20, 0.95);
      color: rgba(240,250,255,0.92);
      text-align: left;
      padding: 14px 12px;
      border-bottom: 1px solid var(--border);
      font-weight: 800;
      letter-spacing: 0.4px;
    }

    tbody td{
      padding: 12px 12px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      color: rgba(240,250,255,0.88);
      vertical-align: middle;
    }

    tbody tr:nth-child(odd){
      background: rgba(255,255,255,0.015);
    }

    tbody tr:hover{
      background: rgba(0, 204, 255, 0.06);
    }

    /* Trust styles */
    .trust-pill{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(0,0,0,0.20);
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }

    .high { color: var(--green); border-color: rgba(53,224,122,0.35); background: rgba(53,224,122,0.08); }
    .med  { color: var(--yellow); border-color: rgba(255,216,77,0.35); background: rgba(255,216,77,0.08); }
    .low  { color: var(--red); border-color: rgba(255,77,77,0.35); background: rgba(255,77,77,0.08); }

    /*arrow */
    .trend-up { 
        color: var(--green); 
        font-weight: 900;       /* Make it bold */
        font-size: 1.2em;       /* Make it larger */
        display: inline-block;
        transform: translateY(1px); /* Align with text */
    }
    
    .trend-down { 
        color: var(--red); 
        font-weight: 900;       /* Make it bold */
        font-size: 1.2em;       /* Make it larger */
        display: inline-block;
        transform: translateY(1px); /* Align with text */
    }

    /* Status pill */
    .status{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.10);
      font-weight: 700;
      letter-spacing: 0.25px;
      font-size: 12.5px;
      background: rgba(0,0,0,0.18);
    }
    .status .dot{
      width: 8px; height: 8px; border-radius: 50%;
      background: rgba(255,255,255,0.4);
    }
    .status.online { border-color: rgba(53,224,122,0.35); color: rgba(53,224,122,0.95); background: rgba(53,224,122,0.08); }
    .status.online .dot { background: var(--green); }

    .status.offline { border-color: rgba(255,77,77,0.35); color: rgba(255,77,77,0.95); background: rgba(255,77,77,0.08); }
    .status.offline .dot { background: var(--red); }

    /* Offline row */
    tr.dead{
      opacity: 0.45;
    }
    tr.dead td{
      color: rgba(240,250,255,0.55);
    }

    .mono{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-variant-numeric: tabular-nums;
    }

    .footer{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12.5px;
    }

    /* Tooltip */
.row-tip{
  position: relative;
}

.row-tip .tip{
  position: absolute;
  left: 12px;
  bottom: 100%;
  top: auto;
  margin-bottom: 10px;
  margin-top: 0;
  width: 360px;
  padding: 12px 12px;
  border-radius: 12px;
  background: rgba(10,14,20,0.98);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 18px 40px rgba(0,0,0,0.45);
  color: rgba(240,250,255,0.92);
  font-size: 12.5px;
  line-height: 1.35;
  z-index: 20;

  opacity: 0;
  transform: translateY(6px);
  pointer-events: none;
  transition: opacity 120ms ease, transform 120ms ease;
}

.row-tip:hover .tip{
  opacity: 1;
  transform: translateY(0);
}

/* small arrow */
.row-tip .tip::before{
  content:"";
  position:absolute;
  top:auto;
  bottom: -6px;
  transform: rotate(225deg);
  left: 18px;
  width: 12px;
  height: 12px;

  background: rgba(10,14,20,0.98);
  border-left: 1px solid rgba(255,255,255,0.10);
  border-top: 1px solid rgba(255,255,255,0.10);
}

.tip-grid{
  display: grid;
  grid-template-columns: 140px 1fr;
  gap: 6px 10px;
}

.tip-k{
  color: rgba(240,250,255,0.60);
}

.tip-v{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  color: rgba(240,250,255,0.92);
  font-variant-numeric: tabular-nums;
}

@media (prefers-reduced-motion: reduce){
  .row-tip .tip{ transition: none; }
}

/*Glowing effect for active nodes */
    tr.active-row td {
      background: rgba(0, 204, 255, 0.15) !important;
      color: #fff !important;
      border-top: 1px solid rgba(0, 204, 255, 0.5);
      border-bottom: 1px solid rgba(0, 204, 255, 0.5);
      transition: background 0.3s ease;
    }

    /* Optional: Make the Node ID pulse */
    tr.active-row .mono {
      color: #00ccff;
      font-weight: bold;
      text-shadow: 0 0 8px rgba(0, 204, 255, 0.6);
    }
  </style>
</head>
<body>
<div class="container">
    <!-- Header / Badge -->
    <div class="topbar">
      <div>
        <div class="lab-title">Autonomous Distributed Systems Lab</div>
        <div class="lab-subtitle">CS @ Umeå University</div>
      </div>

      <div style="text-align:right;">
        <h1 class="page-title">Distributed GenAI Peer Trust Monitor</h1>
        <div class="page-sub">Auto-refreshing every 5s</div>
      </div>
    </div>

    <!-- Registry Panel -->
    <div class="panel">
      <div class="panel-header">
        <h3>Anchor Registry Status</h3>
        <p class="hint">Live view · Trust + latency + liveness</p>
      </div>

      <table>
        <thead>
          <tr>
            <th>Node ID</th>
            <th>IP Address</th>
            <th>Role</th>
            <th>Layers</th>
            <th>Trust Score</th>
            <th>Est. Latency</th>
            <th>Last Seen</th>
            <th>Status</th>
          </tr>
        </thead>

        <tbody>
        {% for id, node in registry.items() %}

          <tr class="{{ 'dead' if not node.alive else '' }} {{ 'active-row' if (now - node.last_active_ts|default(0)) < 20.0 else '' }}">
            <td class="mono row-tip">{{ id }}
            <div class="tip">
    <div class="tip-grid">
      <div class="tip-k">cpu_load</div><div class="tip-v">{{ node.cpu_load }}</div>
      <div class="tip-k">fail_rate</div><div class="tip-v">{{ "%.4f"|format(node.fail_rate|float) }}</div>
      <div class="tip-k">net_delay_ms</div><div class="tip-v">{{ node.net_delay_ms }}</div>
      <div class="tip-k">EWMA latency</div><div class="tip-v">{{ node.latency }} ms</div>
    </div>
  </div>
            </td>
            <td class="mono">{{ node.ip }}:{{ node.port }}</td>
            <td>Worker</td>
            <td class="mono">[{{ node.layer_start }}–{{ node.layer_end }}]</td>

            <td>
              <span class="trust-pill {{ 'high' if node.trust >= 0.8 else 'med' if node.trust >= 0.6 else 'low' }}">
                {{ "%.2f"|format(node.trust) }}
                {% if node.display_trust_trend == "up" %}
                  <span class="trend-up">↑</span>
                {% endif %}
                {% if node.display_trust_trend == "down" %}
                  <span class="trend-down">↓</span>
                {% endif %}
              </span>
            </td>

            <td class="mono">{{ node.latency }} ms</td>
            <td class="mono">{{ node.age_str }}</td>

            <td>
              {% if node.alive %}
                <span class="status online"><span class="dot"></span>ONLINE</span>
              {% else %}
                <span class="status offline"><span class="dot"></span>OFFLINE</span>
              {% endif %}
            </td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>

  </div>
</body>
</html>
"""