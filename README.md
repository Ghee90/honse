# Honse — Spread Smart Dispatcher

Automated dispatcher for the Bahn Express Spread challenge.

## Prerequisites

- **Python 3.10+** (uses standard library only — no `pip install` needed)
- A valid `SPREAD_KEY` from [spread.bahnexpress.com](https://spread.bahnexpress.com)

## Quick Start

1. **Set your team key:**
   ```bash
   export SPREAD_KEY="your-team-key-here"
   ```
   *(Or place `SPREAD_KEY=your-team-key-here` in a `.env` file in the project root.)*

2. **Run a practice week (default: Level 8):**
   ```bash
   python3 dispatch.py
   ```

## Common Commands

- **Practice on a specific level & seed:**
  ```bash
  python3 dispatch.py --level 8 --seed spread-1
  ```
- **Run a 5-week practice pack:**
  ```bash
  python3 dispatch.py --level 8 --pack
  ```
- **Play an official practice ranking run:**
  ```bash
  python3 dispatch.py --level 8 --board
  ```
- **Connect to the live final / test-final:**
  ```bash
  python3 dispatch.py --final
  ```
- **Launch the local dashboard UI:**
  ```bash
  python3 server.py
  ```
