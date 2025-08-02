import struct

class CompressedChessBoard:
    def __init__(self, bbs, stm, enp_sq, rights, halfm, fullm):
        self.bbs = bbs
        self.stm = stm
        self.enp_sq = enp_sq
        self.rights = rights
        self.halfm = halfm
        self.fullm = fullm

class SearchData:
    def __init__(self, best_move, score, visit_distribution):
        self.best_move = best_move
        self.score = score
        self.visit_distribution = visit_distribution

class MontyFormat:
    def __init__(self, startpos, castling, result, moves):
        self.startpos = startpos
        self.castling = castling
        self.result = result
        self.moves = moves

def read_u8(reader): return struct.unpack("<B", reader.read(1))[0]
def read_u16(reader): return struct.unpack("<H", reader.read(2))[0]
def read_u64(reader): return struct.unpack("<Q", reader.read(8))[0]

def decode_move(mv):
    from_sq = (mv >> 6) & 0x3F
    to_sq = mv & 0x3F
    flag = (mv >> 12) & 0xF
    return from_sq, to_sq, flag

def deserialise_from(path):
    with open(path, "rb") as f:
        bbs = [read_u64(f) for _ in range(4)]
        stm = read_u8(f)
        enp_sq = read_u8(f)
        rights = read_u8(f)
        halfm = read_u8(f)
        fullm = read_u16(f)

        compressed = CompressedChessBoard(bbs, stm, enp_sq, rights, halfm, fullm)
        startpos = f"Position(bbs={bbs}, stm={stm})"
        rook_files = [[read_u8(f), read_u8(f)] for _ in range(2)]
        castling = f"Castling(from rook_files={rook_files})"
        result = read_u8(f) / 2.0

        moves = []
        move_num = 0

        while True:
            best_move = read_u16(f)
            if best_move == 0:
                break

            score = read_u16(f) / float(0xFFFF)
            num_moves = read_u8(f)

            visit_distribution = []
            if num_moves > 0:
                visit_distribution = [read_u8(f) for _ in range(num_moves)]

            from_sq, to_sq, flag = decode_move(best_move)

            print(f"Move #{move_num}")
            print(f"  Best move       : {best_move} (from={from_sq}, to={to_sq}, flag={flag})")
            print(f"  Root score (Q)  : {score}")
            print(f"  Num visit moves : {num_moves}")
            if num_moves > 0:
                print(f"  Scaled visit values ({num_moves}): {visit_distribution}")
            print()

            moves.append(SearchData(best_move, score, visit_distribution))
            move_num += 1

# Replace with actual path
deserialise_from("output.bin")
