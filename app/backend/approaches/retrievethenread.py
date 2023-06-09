import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from text import nonewlines

# Cognitive SearchとOpenAIのAPIを直接使用した、シンプルなretrieve-then-readの実装です。これは、最初に
# 検索からトップ文書を抽出し、それを使ってプロンプトを構成し、OpenAIで補完生成する
# (answer)をそのプロンプトで表示します。

# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
# (answer) with that prompt.
class RetrieveThenReadApproach(Approach):

    template = \
"あなたは鉄道に関する質問をサポートする案内アシスタントです。" + \
"質問者が「私」で質問しても、「あなた」を使って質問者を指すようにする。" + \
"次の質問に、以下の出典で提供されたデータのみを使用して答えてください。" + \
"各出典元には、名前の後にコロンと実際の情報があり、回答で使用する各事実には必ず出典名を記載します。" + \
"以下の出典の中から答えられない場合は、「わかりません」と答えてください。" + \
"""

###
Question: 'スタンプラリーはどこでやっていますか'

Sources:
info1.txt: ○このたび、開業１２０周年を記念して５月１３日（土）～２１日（日）の９日間、池袋駅に乗り入れる鉄道四社スタンプ ラリーを開催します。
info2.txt: ≪鉄道四社スタンプラリーin池袋駅の内容≫\n■開業１２０周年を記念し、鉄道四社スタンプラリーｉｎ池袋駅を開催します。イベント期間中、専用台紙に４つの スタンプを集めて台紙をノベルティ引換場所にお持ちいただくと、お好きなノベルティをお一つプレゼントします。

Answer:
5月13日（土）から21日（日）の9日間で、池袋に乗り入れる鉄道四社スタンプラリーを開催します。[info1.txt]イベント期間中、専用台紙に４つの スタンプを集めて台紙をノベルティ引換場所にお持ちいただくと、お好きなノベルティをお一つプレゼントします。[info2.txt]

###
Question: '{q}'?

Sources:
{retrieved}

Answer:
"""

    def __init__(self, search_client: SearchClient, openai_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, q: str, overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q,
                                        filter=filter,
                                        query_type=QueryType.SEMANTIC,
                                        query_language="ja-jp",
                                        query_speller="none",
                                        semantic_configuration_name="default",
                                        top=top,
                                        query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        prompt = (overrides.get("prompt_template") or self.template).format(q=q, retrieved=content)
        completion = openai.Completion.create(
            engine=self.openai_deployment,
            prompt=prompt,
            temperature=overrides.get("temperature") or 0.3,
            max_tokens=1024,
            n=1,
            stop=["\n"])

        return {"data_points": results, "answer": completion.choices[0].text, "thoughts": f"Question:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
