import numpy as np


SCHEME_OFFSETS = {
    "BPSK":  0,
    "QPSK":  2,       # 0 + 2
    "16QAM": 6,       # 2 + 4
    "64QAM": 22,      # 6 + 16
}
TOTAL_CLASSES = 86    # 2 + 4 + 16 + 64

# (i_offset, i_step, q_offset, q_step) for converting amplitude values → table indices
SCHEME_NORMALIZATION = {
    "BPSK":  (1, 2, 0, 1),
    "QPSK":  (1, 2, 1, 2),
    "16QAM": (3, 2, 3, 2),
    "64QAM": (7, 2, 7, 2),
}


def build_lookup_table(symbol_map, i_levels, q_levels, global_offset):
    """Build a 2D numpy lookup table mapping (I, Q) constellation points to global class indices.

    Each scheme occupies a contiguous slice of the shared softmax vector:
        BPSK  → [0,  2)
        QPSK  → [2,  6)
        16QAM → [6,  22)
        64QAM → [22, 86)

    The stored value is bits + global_offset, so the index is valid directly
    against the full 86-class softmax output.

    Constellation values are mapped to array indices via:
        idx = (val + offset) // step
    where offset = -levels[0] and step = levels[1] - levels[0] (defaulting to
    1 when only a single level exists, as in the BPSK Q axis).

    Args:
        symbol_map: Dict mapping (i_val, q_val) tuples to integer bit patterns.
        i_levels: Tuple of valid I-axis amplitude values in ascending order.
        q_levels: Tuple of valid Q-axis amplitude values in ascending order.
        global_offset: Integer offset (from SCHEME_OFFSETS) to shift local bit
            pattern values into the global class index space.

    Returns:
        np.ndarray of shape (len(i_levels), len(q_levels)) with dtype uint8,
        where table[i_idx, q_idx] is the global class index for that point.
    """
    i_offset = -i_levels[0]
    i_step = (i_levels[1] - i_levels[0]) if len(i_levels) > 1 else 1
    q_offset = -q_levels[0]
    q_step = (q_levels[1] - q_levels[0]) if len(q_levels) > 1 else 1

    table = np.zeros((len(i_levels), len(q_levels)), dtype=np.uint8)
    for (i, q), bits in symbol_map.items():
        table[(i + i_offset) // i_step, (q + q_offset) // q_step] = bits + global_offset
    return table


def _build_global_index_to_symbol():
    """Build the global array mapping each class index to its binary symbol string.

    Indices 0–1 are BPSK symbols ("0", "1"), 2–5 are QPSK ("00"–"11"),
    6–21 are 16QAM ("0000"–"1111"), and 22–85 are 64QAM ("000000"–"111111").
    All strings are stored with dtype '<U6' (width of the longest symbol).

    Returns:
        np.ndarray of shape (86,) with dtype '<U6'.
    """
    entries = (
        [(i, 1) for i in range(2)]    # BPSK
        + [(i, 2) for i in range(4)]  # QPSK
        + [(i, 4) for i in range(16)] # 16QAM
        + [(i, 6) for i in range(64)] # 64QAM
    )
    return np.array([format(i, f'0{n}b') for i, n in entries], dtype='<U6')


index_to_symbol = _build_global_index_to_symbol()


scheme_to_high_low_map = {
    "BPSK": [(-1, 1), (0,1)],
    "QPSK": [(-1, 1), (-1, 1)],
    "16QAM": [(-2, 2), (-2, 2)],
    "64QAM": [(-4, 4), (-4, 4)],
}

_bpsk_i_levels = (-1, 1)
_bpsk_q_levels = (0,)
bpsk_map = {
    (-1, 0): 0b0,
    (1, 0): 0b1,
}
bpsk_table = build_lookup_table(bpsk_map, _bpsk_i_levels, _bpsk_q_levels, SCHEME_OFFSETS["BPSK"])

_qpsk_levels = (-1, 1)
qpsk_map = {
    (-1, -1): 0b00,
    (-1, 1): 0b01,
    (1, 1): 0b11,
    (1, -1): 0b10,
}
qpsk_table = build_lookup_table(qpsk_map, _qpsk_levels, _qpsk_levels, SCHEME_OFFSETS["QPSK"])

_qam16_levels = (-3, -1, 1, 3)
_qam16_gray = (0b00, 0b01, 0b11, 0b10)
qam16_map = {
    (i, q): (i_bits << 2) | q_bits
    for i, i_bits in zip(_qam16_levels, _qam16_gray)
    for q, q_bits in zip(_qam16_levels, _qam16_gray)
}
qam16_table = build_lookup_table(qam16_map, _qam16_levels, _qam16_levels, SCHEME_OFFSETS["16QAM"])

_qam64_levels = (-7, -5, -3, -1, 1, 3, 5, 7)
_qam64_gray = (0b000, 0b001, 0b011, 0b010, 0b110, 0b111, 0b101, 0b100)
qam64_map = {
    (i, q): (i_bits << 3) | q_bits
    for i, i_bits in zip(_qam64_levels, _qam64_gray)
    for q, q_bits in zip(_qam64_levels, _qam64_gray)
}
qam64_table = build_lookup_table(qam64_map, _qam64_levels, _qam64_levels, SCHEME_OFFSETS["64QAM"])

scheme_to_index_table_map = {
    "BPSK": bpsk_table,
    "QPSK": qpsk_table,
    "16QAM": qam16_table,
    "64QAM": qam64_table,
}

