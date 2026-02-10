# G-TRAC: Trust-Aware Routing for Distributed Generative AI

**G-TRAC** (Generative Trust-Aware Routing and Adaptive Chaining) is a coordination framework for distributed Large Language Model (LLM) inference over decentralized and unreliable edge networks. It transforms inference routing from a best-effort transport problem into a **Risk-Bounded Shortest Path** control problem, ensuring reliable token generation even on heterogeneous devices.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch (CPU or CUDA enabled)
- Hugging Face Transformers

### Steps

1. Clone the repository:
   ```bash
   git clone [https://github.com/anonymous-123qh/g-trac.git](https://github.com/anonymous-123qh/g-trac.git)
   cd g-trac

2. Install packages:
   ```bash
   pip install -r requirements.txt
    ```

## Usage
### Environment Variables

Set the following environment variables before starting any component:

```bash
export TARP_ENGINE=real
export TARP_MODEL=gpt2-large # Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
# Optional: To use GPU
export TARP_DEVICE=cuda
```
1. Start the Anchor
   
**Usage**
```bash
python run.py anchor <port>
```
**Example**
```bash
python run.py anchor 5000
```
2. Start worker
   
**Usage**
```bash
python run.py worker <IP> <PORT> <CPU> <FAIL_RATE> <TRUST> <ID> <ANCHOR_IP> <ANCHOR_PORT> <L_START> <L_END>
```
- `CPU`: CPU burn level (e.g., `1`)
- `FAIL_RATE`: Failure rate in range `[0, 1]`
- `TRUST`: Initial trust score in range `[0, 1]`
  
**Example**  
Starting a worker on port `6001` handling layers `0–35` (monolithic mode for testing):

```bash
python run.py worker 127.0.0.1 6001 0 0 1 worker1 127.0.0.1 5000 0 35
```

   
4. Start client

**Usage**
```bash
python run.py client <ANCHOR_IP> <ANCHOR_PORT> [MODE]
```
MODE (optional): Routing mode to use. Supported values:

- `sp` — Shortest Path (Latency only)

- `mr` — Max Reliability

- `larac` — Lagrangian relaxation

- `naive` — Baseline random routing

- `g-trac` — Trust-aware G-TRAC routing (Default)
  
**Example**   
```bash
python run.py client 127.0.0.1 5000
```
## Visualization

### Real-Time Dashboard
The G-TRAC Anchor provides a web-based dashboard to monitor network health, trust updates, and routing decisions in real-time.

<p align="center">
  <img src="images/screen.png" width="100%" alt="G-TRAC Dashboard" />
</p>
