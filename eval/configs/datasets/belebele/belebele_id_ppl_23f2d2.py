from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import BelebeleDataset, BelebeleEvaluator


belebele_reader_cfg = dict(
    input_columns=['flores_passage', 'question', 'mc_answer1', 'mc_answer2', 'mc_answer3', 'mc_answer4'],
    output_column='answer')

system_prompt = '''Silakan ikuti contoh yang diberikan, baca konteksnya dan jawab pertanyaannya.\n'''

exp_1 = '''Konteks: Simbol nuklir digunakan untuk menulis persamaan nuklir peluruhan radioaktif. Mari kita perhatikan contoh peluruhan beta-minus dari thorium-234 menjadi protaktinium-234. Reaksi ini diwakili oleh persamaan :.
Pertanyaan: Apa yang digunakan untuk menulis persamaan nuklir peluruhan radioaktif?'''
ans_1 = '''Jawaban: simbol nuklir\n'''


exp_2 = '''Konteks: Barang obral apa pun yang dibeli dapat dikembalikan untuk mendapatkan kredit toko tetapi tidak untuk pengembalian uang sebesar harga pembelian. Setiap peralatan rumah tangga dan peralatan berkebun dijual bersama dengan peralatan konstruksi pilihan.
Pertanyaan: Jika pernyataan di atas benar, manakah pernyataan berikut yang juga benar?'''
ans_2 = '''Jawaban: Peralatan berkebun tidak dapat dikembalikan untuk mendapatkan pengembalian uang.\n'''


exp_3 = '''Konteks: Saya punya tujuh tas. Tiga tas berukuran besar, dan empat tas lainnya berukuran kecil. Saya punya beberapa bola basket dan bola voli. Saya memasukkan dua bola voli ke dalam setiap tas kecil. Dan saya menaruh dua bola basket dan dua bola voli di setiap tas besar. Jumlah bola voli adalah usia saya.
Pertanyaan: Dimana bola basketnya?'''
ans_3 = '''Jawaban: Di dalam tas besar.\n'''


prompt_input = '''Konteks: {flores_passage}
Pertanyaan: {question}'''
prompt_output = '''Jawaban:'''


belebele_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            chr(ans+64): dict(
                begin=[
                    dict(role='SYSTEM', fallback_role='HUMAN', prompt=system_prompt),
                ],
                round=[
                    dict(role='HUMAN', prompt=exp_1),
                    dict(role='BOT', prompt=ans_1),
                    dict(role='HUMAN', prompt=exp_2),
                    dict(role='BOT', prompt=ans_2),
                    dict(role='HUMAN', prompt=exp_3),
                    dict(role='BOT', prompt=ans_3),
                    dict(role='HUMAN', prompt=prompt_input),
                    dict(role='BOT', prompt=f'Jawaban: {{mc_answer{ans}}}'),
                ])
            for ans in range(1, 5)
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer, generation_kwargs=dict(do_sample=False)))


belebele_eval_cfg = dict(
    evaluator=dict(type=BelebeleEvaluator),
    pred_role="BOT")

belebele_datasets = [
    dict(
        type=BelebeleDataset,
        abbr='belebele-id-ppl',
        path='./data/belebele/ind_Latn.jsonl',
        reader_cfg=belebele_reader_cfg,
        infer_cfg=belebele_infer_cfg,
        eval_cfg=belebele_eval_cfg)
]
