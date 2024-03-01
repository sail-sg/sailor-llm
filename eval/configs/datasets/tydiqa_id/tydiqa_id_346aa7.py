from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TydiqaIdDataset, TydiqaIdEvaluator


tydiqa_reader_cfg = dict(
    input_columns=['context', 'question'],
    output_column='answer')

system_prompt = '''Baca konteks yang diberikan dan jawab pertanyaannya. Tanggapan Anda harus dalam bahasa Indonesia dan sesederhana mungkin.\n'''

exp_1 = '''Konteks: Komposisi tubuh dapat dianalisis dalam hal jenis molekulnya, misalnya, air, protein, jaringan ikat, lemak (atau lipida), hidroksilapatit (dalam tulang), karbohidrat (seperti glikogen dan glukosa) dan DNA. Sesuai jenis jaringannya, tubuh dapat dianalisis menjadi air, lemak, otot, tulang, dll. Menurut jenis selnya, tubuh mengandung ratusan jenis sel, tetapi tercatat jumlah sel paling banyak yang dikandung dalam tubuh manusia (meski bukan massa sel terbesar) bukan sel manusia, tetapi bakteri yang berada dalam saluran pencernaan manusia normal.
Pertanyaan: Ada berapa banyak sel dalam tubuh manusia?'''
ans_1 = '''Jawaban: ratusan jenis sel\n'''

exp_2 = '''Konteks: Restoran pertama Burger King dinamai Insta Burger King dan dibuka pada tahun 1954 di Miami, Florida, Amerika Serikat oleh James McLamore dan David Edgerton, keduanya adalah alumni dari Cornell University School of Hotel Administration.
Pertanyaan: Siapakah pendiri Burger King?'''
ans_2 = '''Jawaban: James McLamore dan David Edgerton\n'''

exp_3 = '''Konteks: Love Nikki-Dress Up Queen atau dikenal di Indonesia sebagai Love Nikki-Dress Up Fantasy atau menyebut Miracle Nikki, adalah permainan video berbasis ponsel yang dibuat oleh SuZhou Nikki Co. dan diterbitkan oleh Elex Technology. Premis dari permainan ini adalah untuk menata karakter utama Nikki dan menantang NPC dan pemain dalam gaya duel, di mana peserta dengan selera mode terbaik dan pemenang yang paling relevan.
Pertanyaan: siapakah pencipta game Love Nikki-Dress Up Queen?'''
ans_3 = '''Jawaban: SuZhou Nikki Co\n'''

prompt_input = '''Konteks: {context}
Pertanyaan: {question}'''
prompt_output = '''Jawaban:'''

tydiqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
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
                dict(role='BOT', prompt=prompt_output),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, generation_kwargs=dict(do_sample=False), batch_size=4, max_out_len=100))

tydiqa_eval_cfg = dict(
    evaluator=dict(type=TydiqaIdEvaluator),
    pred_role="BOT")

tydiqa_datasets = [
    dict(
        type=TydiqaIdDataset,
        abbr='tydiqa-id',
        path='./data/tydiqa_id/validation.jsonl',
        reader_cfg=tydiqa_reader_cfg,
        infer_cfg=tydiqa_infer_cfg,
        eval_cfg=tydiqa_eval_cfg)
]