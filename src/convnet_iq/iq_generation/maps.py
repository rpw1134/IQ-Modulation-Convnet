bpsk_map = {
    (-1, 0): 0b0,
    (1, 0): 0b1,
}

qpsk_map = {
    (-1, -1): 0b00,
    (-1, 1): 0b01,
    (1, 1): 0b11,
    (1, -1): 0b10,
}

_qam16_levels = (-3, -1, 1, 3)
_qam16_gray = (0b00, 0b01, 0b11, 0b10)
qam16_map = {
    (i, q): (i_bits << 2) | q_bits
    for i, i_bits in zip(_qam16_levels, _qam16_gray)
    for q, q_bits in zip(_qam16_levels, _qam16_gray)
}

_qam64_levels = (-7, -5, -3, -1, 1, 3, 5, 7)
_qam64_gray = (0b000, 0b001, 0b011, 0b010, 0b110, 0b111, 0b101, 0b100)
qam64_map = {
    (i, q): (i_bits << 3) | q_bits
    for i, i_bits in zip(_qam64_levels, _qam64_gray)
    for q, q_bits in zip(_qam64_levels, _qam64_gray)
}

