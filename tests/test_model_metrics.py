from wynxo.provider import Chunk


def test_chunk_uses_load_duration_not_total_duration():
    chunk = Chunk(total_duration_ns=900, load_duration_ns=120)
    assert chunk.load_duration_ns == 120
    assert chunk.total_duration_ns == 900
