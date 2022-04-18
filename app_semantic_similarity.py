import re
import io
import nltk
import logging
import pandas as pd
from flask import Flask
# import numpy as np
# from flask import jsonify
from flask import Response
from flask_restplus import Api, Resource
from werkzeug.datastructures import Headers
from werkzeug.datastructures import FileStorage
from embeddings import Embedding, transformer_func
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from waitress import serve


def api_configurator(name_space):
    """"""
    upload_parser = name_space.parser()
    upload_parser.add_argument("text1", type=FileStorage, location='files', required=True,
                               help="utf8 encoded text file")
    upload_parser.add_argument("text2", type=FileStorage, location='files', required=True,
                               help="utf8 encoded text file")
    # upload_parser.add_argument("score", type=float, required=False)
    return upload_parser


def response_func(clustering_texts_df, response_type="excel"):
    """"""
    headers = Headers()
    if response_type == "excel":
        headers.add('Content-Disposition', 'attachment', filename="similarity_sentences.xlsx")
        buffer = io.BytesIO()
        clustering_texts_df.to_excel(buffer, index=False, encoding='cp1251'),
        return Response(buffer.getvalue(),
                        mimetype='application/vnd.ms-excel',
                        headers=headers)

    else:
        headers.add('Content-Disposition', 'attachment', filename="similarity_sentences.csv")
        return Response(clustering_texts_df.to_csv(index=False),
                        mimetype="text/csv",
                        headers=headers)

nltk.download('punkt')
logger = logging.getLogger("app_similarity")
logger.setLevel(logging.DEBUG)


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

name_space = api.namespace('api', 'загрузка двух текстовых файлов, возвращает коэффициент "похожести" и'
                                  ' excel файл с самыми похожими предложениями')
upload_parser = api_configurator(name_space)

transformer_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embedder = Embedding(transformer_model, transformer_func)


@name_space.route('/')
@name_space.expect(upload_parser)
class TextsSimilarity(Resource):
    def post(self):
        """POST method on input csv file with texts and score, output clustering texts  as csv file."""
        args = upload_parser.parse_args()
        tx_code1 = args["text1"].read()
        tx_code2 = args["text2"].read()
        tx1 = tx_code1.decode('utf8')
        tx2 = tx_code2.decode('utf8')

        tx1_ = re.sub("\n", " ", tx1)
        tx2_ = re.sub("\n", " ", tx2)
        sp_tx1 = nltk.sent_tokenize(tx1_, language="russian")
        sp_tx2 = nltk.sent_tokenize(tx2_, language="russian")

        text1_vectors = embedder(sp_tx1)
        text2_vectors = embedder(sp_tx2)
        similarity_matrix = cosine_similarity(text1_vectors, text2_vectors)
        sentences_vectors = []
        for vs, s1 in zip(similarity_matrix, sp_tx1):
            s2_sc = sorted([(v, s2) for v, s2 in zip(vs, sp_tx2)], key=lambda x: x[0], reverse=True)[0]
            sentences_vectors.append((s1, s2_sc[1], s2_sc[0]))

        sentences_vectors_df = pd.DataFrame(sentences_vectors, columns=["sentence1", "sentence2", "score"])
        # result = [max(v) for v in similarity_matrix]
        # return jsonify({"result:": str(np.mean(result))})
        return response_func(sentences_vectors_df)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    # serve(app, host="0.0.0.0", port=8080)
