"""Execute ordinary Python cells and save notebook outputs without installing packages."""
from pathlib import Path
import ast
import base64
import contextlib
import io
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import IPython.display

path = ROOT / 'main_copy.ipynb'
nb = json.loads(path.read_text(encoding='utf-8'))
namespace = {'__name__': '__main__'}
outputs = []
capture = io.StringIO()

def flush_text():
    value = capture.getvalue()
    if value:
        outputs.append({'output_type': 'stream', 'name': 'stdout',
                        'text': value.splitlines(keepends=True)})
        capture.seek(0)
        capture.truncate(0)

def display(*objects, **kwargs):
    flush_text()
    for obj in objects:
        data = {'text/plain': repr(obj)}
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
            data['text/html'] = frame.to_html(max_rows=40, max_cols=16)
        elif hasattr(obj, '_repr_html_'):
            value = obj._repr_html_()
            if value is not None:
                data['text/html'] = value
        outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': data})

def show(*args, **kwargs):
    flush_text()
    for number in plt.get_fignums():
        fig = plt.figure(number)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
        outputs.append({'output_type': 'display_data', 'metadata': {},
                        'data': {'image/png': base64.b64encode(buffer.getvalue()).decode('ascii'),
                                 'text/plain': repr(fig)}})
        plt.close(fig)

IPython.display.display = display
plt.show = show
namespace['display'] = display
cell_prefix = sys.argv[1] if len(sys.argv) > 1 else None
execution = max((c.get('execution_count') or 0 for c in nb['cells']
                 if c['cell_type'] == 'code' and cell_prefix
                 and not c.get('id', '').startswith(cell_prefix)), default=0)
executed_this_run = 0
for index, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    if cell_prefix and not cell.get('id', '').startswith(cell_prefix):
        continue
    execution += 1
    executed_this_run += 1
    outputs = []
    source = ''.join(cell['source'])
    print(f'Executing cell {index}: {cell.get("id", "")}', flush=True)
    try:
        tree = ast.parse(source, filename=f'notebook-cell-{index}')
        tail = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tail = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            exec(compile(tree, f'notebook-cell-{index}', 'exec'), namespace)
            if tail is not None:
                result = eval(compile(tail, f'notebook-cell-{index}', 'eval'), namespace)
                if result is not None:
                    display(result)
            flush_text()
        cell['outputs'] = outputs
        cell['execution_count'] = execution
    except Exception:
        flush_text()
        cell['outputs'] = outputs
        traceback.print_exc()
        # Do not save a partially executed notebook over the edited file.
        sys.exit(1)

if 'df1' in namespace:
    assert namespace['coverage_map'].to_numpy().sum() == len(namespace['map_points'])
    assert namespace['target_afr_map'].notna().equals(namespace['target_support_map'])
    assert namespace['measured_afr_map'].notna().equals(namespace['measured_support_map'])
    assert len(namespace['df_raw']) == len(namespace['df1'])
    assert namespace['target_afr_map'].notna().to_numpy().any()
    assert namespace['measured_afr_map'].notna().to_numpy().any()
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
summary = {
    'code_cells_executed_this_run': executed_this_run,
    'total_saved_executed_cells': sum(c.get('execution_count') is not None for c in nb['cells'] if c['cell_type']=='code'),
}
if 'multi_result' in namespace:
    multi = namespace['multi_result']
    summary['all_trips'] = {
        'files': len(multi['trips']),
        'raw_rows': int(multi['trips'].raw_rows.sum()),
        'aligned_rows': int(multi['trips'].grid_rows.sum()),
        'supported_cells': {
            metric: int(matrix.notna().sum().sum()) for metric, matrix in multi['maps'].items()
        },
        'output_dir': str(multi['output_dir'].resolve()),
    }
if 'linear_result' in namespace:
    linear = namespace['linear_result']
    summary['linear_completion'] = linear['coverage'].to_dict(orient='records')
    summary['linear_output_dir'] = str(linear['output_dir'].resolve())
(ROOT / '_notebook_work' / 'execution_summary.json').write_text(
    json.dumps(summary, indent=2), encoding='utf-8'
)
print(json.dumps(summary, indent=2))
