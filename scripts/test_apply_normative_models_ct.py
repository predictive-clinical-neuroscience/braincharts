from testbook import testbook

@testbook('apply_normative_models_ct.ipynb', execute=True)
def test_stdout(tb):
    assert tb.cell_output_text(1) == 'hello world!'