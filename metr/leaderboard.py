import gradio as gr
import polars as pl

eval =  pl.scan_csv('storage/evaluation.csv')
datasets = pl.scan_csv('storage/datasets.csv')


def eval_domains(bm25: bool, k: int) -> pl.DataFrame:
    metrics = ['precision', 'recall', 'f1', 'mrr']

    return (
        eval.filter(bm25=bm25, k=k)
        .join(datasets, on='dataset')
        .group_by('model', 'domain').agg(
            (
                pl.col(metric).dot('rows_k') / pl.col('rows_k').sum()
            ).round(2)
            for metric in metrics
        ).with_columns(
            rank=pl.mean_horizontal(metrics).rank().over('domain')
        )
        .select(
            'model', 'domain', 'rank', *metrics
        )
        .collect()
    )


with gr.Blocks() as demo:
    gr.Markdown('# METR')
    with gr.Tab("Per domain"):
        per_domain=gr.DataFrame()
    with gr.Tab("Per dataset") as second_tab:
        gr.Button("New Tiger")
    
    with gr.Row(equal_height=True):
        with gr.Column():
            pass
        with gr.Column():
            bm25 = gr.Checkbox(label='BM25', scale=0)
            k = gr.Dropdown(label='k', choices=[1], scale=0)
    
    bm25.change(eval_domains, inputs=[bm25, k], outputs=per_domain)
    k.change(eval_domains, inputs=[bm25, k], outputs=per_domain)
    
    demo.load(eval_domains, inputs=[bm25, k], outputs=per_domain)
    #second_tab.select(eval_domains, inputs=[bm25, k], outputs=per_domain)
    
    gr.Markdown('# Благодарности')
    

demo.launch()
