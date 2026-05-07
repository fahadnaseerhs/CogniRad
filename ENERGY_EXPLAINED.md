# 🔋 Student Energy Accumulation - Explained

## What is "Energy" in CogniRad?

**Energy** is a numerical score (measured in Joules) that represents **how much a student is contributing to channel congestion**.

Think of it like this:
- **Real WiFi**: Every device transmitting uses radio frequency (RF) power
- **CogniRad**: We simulate this by tracking "energy" per student
- **Higher energy** = More stress on the channel

---

## 📊 What Does Energy Represent in Real Life?

### Real-World WiFi Physics

In actual WiFi networks, every transmission:

1. **Consumes RF Power** (measured in milliwatts or dBm)
   - Sending data requires electromagnetic energy
   - More data = more energy radiated into the air

2. **Occupies Airtime** (channel utilization)
   - WiFi is a shared medium (like a conversation)
   - When you transmit, others must wait
   - High utilization = congestion

3. **Creates Interference**
   - Multiple devices transmitting simultaneously
   - Signals overlap and degrade each other
   - Reduces Signal-to-Noise Ratio (SNR)

4. **Degrades Modulation**
   - Clean channel: 64-QAM (6 bits/symbol) - fast
   - Noisy channel: BPSK (1 bit/symbol) - slow
   - More interference = slower data rates

### CogniRad's Simulation

We model all of this with a single **energy score per student**:

```
Energy = Base Energy 
       + (Message Size × Energy per Bit)
       + (Channel Utilization × Weight)
       + (Contention from Other Users × Weight)
```

**Example**:
- Short message "hi" on empty channel: **~0.08 J**
- Long message (500 chars) on busy channel: **~2.5 J**
- Rapid burst of messages: **~5-10 J** (high bitrate)

---

## 🔄 How Energy Accumulates

### When a Student Sends a Message

```python
# 1. Calculate message energy based on:
#    - Message length (more chars = more energy)
#    - Time since last message (fast bursts = high bitrate)
#    - Number of users on channel (contention penalty)
#    - Channel band (2.4 GHz vs 5 GHz)

msg_energy = compute_phy_event(
    text="Hello, how are you?",
    channel_id="CH-1",
    dt_seconds=0.5,  # 500ms since last message (fast!)
    n_users=5
)
# Result: msg_energy ≈ 0.15 J

# 2. Add to student's cumulative total
student_total = 2.3 J  # previous total
student_total += 0.15 J
# New total: 2.45 J
```

### Energy Grows When:

✅ **Sending long messages** (more bits to transmit)
✅ **Sending messages rapidly** (high bitrate → high utilization)
✅ **Many users on same channel** (contention penalty)
✅ **Using 2.4 GHz band** (noisier, more energy needed)

### Energy Decreases When:

✅ **Student is idle** (automatic decay every second)
✅ **Student is reallocated** (50% decay on move)
✅ **Fewer active users** (faster decay rate)

---

## ⏱️ Idle Decay - Natural Recovery

Energy doesn't stay forever. It **decays exponentially** when students are idle.

### Decay Formula

```
New Energy = Old Energy × Decay Factor
```

**Decay Factor** is dynamic (scales with total active students):

| Active Students | Decay Factor (per second) | Half-Life |
|-----------------|---------------------------|-----------|
| 0-5 students    | 0.979 (fast decay)        | ~33 sec   |
| 10-20 students  | 0.985 (medium decay)      | ~46 sec   |
| 30-50 students  | 0.990 (slow decay)        | ~69 sec   |

**Why dynamic?**
- **Few users**: Channel should clear quickly (fast decay)
- **Many users**: Busy network stays energized longer (slow decay)

### Example Decay Timeline

```
Student sends burst of messages → Energy = 10.0 J
After 10 seconds idle:  10.0 × 0.979^10 = 8.1 J
After 30 seconds idle:  10.0 × 0.979^30 = 5.3 J
After 60 seconds idle:  10.0 × 0.979^60 = 2.8 J
After 120 seconds idle: 10.0 × 0.979^120 = 0.8 J
```

---

## 🚨 What Happens When Energy Gets High?

### Channel Status Thresholds

The system uses **dynamic thresholds** that scale with the number of users:

```
Threshold = Coefficient × sqrt(N)
```

| Status | Coefficient | Example (N=10) | What It Means |
|--------|-------------|----------------|---------------|
| **FREE** | 3.0 | < 9.5 J | Channel is healthy |
| **BUSY** | 10.0 | < 31.6 J | Active but not stressed |
| **CONGESTED** | 20.0 | < 63.2 J | Overloaded, reallocation needed |
| **JAMMED** | 35.0 | < 110.7 J | Critical, immediate action |

### What Happens at Each Level

#### 🟢 FREE (Total Energy < 9.5 J for 10 users)
- ✅ All messages accepted
- ✅ Low latency (~0-50 ms)
- ✅ High modulation (64-QAM)
- ✅ No action needed

#### 🟡 BUSY (9.5 J - 31.6 J)
- ✅ Messages still accepted
- ⚠️ Moderate latency (~50-200 ms)
- ⚠️ Medium modulation (16-QAM)
- ℹ️ System monitors closely

#### 🟠 CONGESTED (31.6 J - 63.2 J)
- ⚠️ Some messages may be dropped (10-60% drop rate)
- ⚠️ High latency (~200-1000 ms)
- ⚠️ Low modulation (QPSK)
- 🔄 **Automatic reallocation triggered**

#### 🔴 JAMMED (> 63.2 J)
- ❌ Most messages dropped (60-100% drop rate)
- ❌ Extreme latency (~1000-2500 ms)
- ❌ Minimum modulation (BPSK)
- 🚨 **Emergency evacuation of all users**

---

## 🔄 Reallocation Process

When a channel becomes **CONGESTED** or **JAMMED**, the system automatically reallocates students.

### Step-by-Step

1. **Detect Congestion**
   ```
   Channel CH-1: Total Energy = 65 J (CONGESTED)
   Users: [CMS001: 15J, CMS002: 12J, CMS003: 10J, ...]
   ```

2. **Sort by Energy** (highest first)
   ```
   Candidates: CMS001 (15J), CMS002 (12J), CMS003 (10J), ...
   ```

3. **Apply Round-Robin Fairness**
   - Don't always pick the same student
   - Rotate starting position each time

4. **Find Valid Destination**
   - Check each channel (skip JAMMED)
   - Project: "If I move CMS001 (15J × 0.5 decay = 7.5J) to CH-3..."
   - Verify CH-3 stays healthy after absorbing 7.5J
   - Accept only if destination remains FREE or BUSY

5. **Execute Move**
   ```
   Move CMS001: CH-1 → CH-3
   Energy decays: 15J → 7.5J (50% on relocation)
   ```

6. **Reclassify Source**
   ```
   CH-1 new total: 65J - 15J = 50J (still CONGESTED)
   Continue moving students...
   ```

7. **Stop When Healthy**
   ```
   CH-1 new total: 28J (BUSY - healthy!)
   Stop reallocation (minimum-move principle)
   ```

---

## 📈 Real-World Analogy

Think of a **coffee shop WiFi**:

### Scenario 1: Light Load
- 3 people browsing web
- **Energy**: Each person ~2 J
- **Total**: 6 J (FREE)
- **Result**: Fast, responsive WiFi

### Scenario 2: Heavy Load
- 15 people streaming video
- **Energy**: Each person ~8 J
- **Total**: 120 J (JAMMED)
- **Result**: Buffering, timeouts, dropped connections

### CogniRad's Solution
- Detect the congestion (120 J > threshold)
- Move 8 people to a different frequency band
- **New totals**: 
  - Original band: 56 J (BUSY - healthy)
  - New band: 32 J (BUSY - healthy)
- **Result**: Everyone gets good service

---

## 🎯 Why This Matters for Your Presentation

### Key Points to Emphasize

1. **Energy = Real RF Physics**
   - Not arbitrary numbers
   - Models actual WiFi behavior
   - Based on IEEE 802.11 standards

2. **Dynamic Thresholds = Smart Scaling**
   - Works for 1 user or 50 users
   - Adapts to actual load
   - Prevents false positives

3. **Decay = Natural Recovery**
   - Channels heal themselves during idle periods
   - No manual intervention needed
   - Mimics real RF propagation

4. **Reallocation = Cognitive Radio**
   - Automatic load balancing
   - Fair (round-robin selection)
   - Efficient (minimum moves)

### Demo Talking Points

**When showing energy accumulation:**
> "Each message a student sends adds energy to their channel. This simulates real WiFi transmission power. Watch as rapid messaging causes energy to spike."

**When showing decay:**
> "Notice how energy decreases when students are idle. This models how radio waves dissipate over time. The channel naturally recovers."

**When triggering reallocation:**
> "The total energy just crossed the congestion threshold. The system automatically moves the highest-energy students to healthier channels. Their energy is cut in half during the move, simulating the cost of reestablishing a connection."

**When showing dynamic thresholds:**
> "The thresholds aren't fixed. With 5 users, the congestion limit is 44 Joules. With 30 users, it's 109 Joules. This square-root scaling mirrors how real WiFi handles contention."

---

## 🧪 Try It Yourself

### Experiment 1: Energy Accumulation
```bash
# Student sends 10 rapid messages
# Watch energy climb: 0 → 2 → 4 → 6 → 8 → 10 J
# Channel status: FREE → BUSY → CONGESTED
```

### Experiment 2: Idle Decay
```bash
# Student stops sending
# Watch energy decay: 10 → 8 → 6 → 4 → 2 → 0 J
# Channel status: CONGESTED → BUSY → FREE
```

### Experiment 3: Reallocation
```bash
# 5 students send rapid messages on CH-1
# Total energy: 50 J (CONGESTED)
# System moves 2 students to CH-2
# CH-1: 50 → 30 J (BUSY)
# CH-2: 0 → 10 J (BUSY)
```

---

## 📚 Technical Details

### Energy Calculation (Full Formula)

```python
msg_energy = (
    base_energy                          # 0.08 J (2.4 GHz) or 0.06 J (5 GHz)
    + energy_per_bit × bits              # 1.5e-5 J/bit × (chars × 8)
    + utilization_weight × utilization   # 0.5 × (bitrate / capacity)
    + contention_weight × contention     # 0.10 × (N-1)/sqrt(N)
)
```

### Decay Calculation (Full Formula)

```python
# Dynamic decay factor
ratio = min(n_active_students / 50, 1.0)
decay_factor = 0.979 + (0.990 - 0.979) × ratio

# Apply decay
elapsed_ticks = (now - last_update) / 1.0  # 1-second ticks
new_energy = old_energy × (decay_factor ** elapsed_ticks)

# Clamp to zero
if new_energy < 0.01:
    new_energy = 0.0
```

### Threshold Calculation (Full Formula)

```python
n = number_of_users_on_channel

FREE_threshold      = 3.0  × sqrt(n)
BUSY_threshold      = 10.0 × sqrt(n)
CONGESTED_threshold = 20.0 × sqrt(n)
JAMMED_threshold    = 35.0 × sqrt(n)
```

---

## ✅ Summary

| Concept | Real-World Equivalent | CogniRad Implementation |
|---------|----------------------|-------------------------|
| **Energy** | RF transmission power | Joules per student |
| **Accumulation** | Repeated transmissions | Add energy per message |
| **Decay** | Signal dissipation | Exponential decay (0.979-0.990/sec) |
| **Thresholds** | Channel capacity limits | Dynamic sqrt(N) scaling |
| **Reallocation** | Frequency handoff | Move students to healthier channels |
| **Fairness** | Equal access | Round-robin selection |

**Bottom Line**: Energy is a realistic simulation of WiFi physics that enables intelligent, automatic spectrum management.
