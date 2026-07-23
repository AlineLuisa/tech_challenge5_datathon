# -*- coding: utf-8 -*-
"""
app.py — Passos Mágicos | Risco de Defasagem Escolar
-----------------------------------------------------
Aplicação Streamlit que disponibiliza o modelo preditivo treinado no notebook
`passos_magicos_analise_completa.ipynb` (Pergunta 9) como uma ferramenta de
priorização de apoio pedagógico.

O modelo (Random Forest dentro de um Pipeline de pré-processamento) foi
treinado para responder: "dados os indicadores do aluno no ano corrente (t),
qual a probabilidade de ele estar defasado no ano seguinte (t+1)?"

Como rodar:
    streamlit run app.py

Requisitos:
    - O arquivo do modelo treinado (ex.: modelo_risco.pkl, gerado via
      joblib.dump no notebook) precisa estar na MESMA pasta deste app.py.
    - streamlit, pandas, numpy, scikit-learn e joblib instalados
      (ver requirements.txt).
"""

import os
import re

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ----------------------------------------------------------------------------
# Configuração geral
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Passos Mágicos — Risco de Defasagem",
    page_icon="🎯",
    layout="wide",
)

MODEL_PATH = "modelo_risco.pkl"

# Mapeamentos usados na Parte 2 do notebook (feature engineering) — precisam
# ser EXATAMENTE os mesmos usados no treino, ou o modelo interpreta errado
# os dados de entrada.
FASE_MAP = {
    "ALFA (pré-inicial / alfabetização)": 0,
    "Fase 1": 1,
    "Fase 2": 2,
    "Fase 3": 3,
    "Fase 4": 4,
    "Fase 5": 5,
    "Fase 6": 6,
    "Fase 7": 7,
}

PEDRA_MAP = {"Quartzo": 1, "Ágata": 2, "Ametista": 3, "Topázio": 4}

INSTITUICAO_MAP = {
    "Escola pública": "Pública",
    "Escola privada": "Privada",
    "Concluiu o ensino / cursando universidade": "Concluiu/Universitário",
    "Outra / não informado": "Outra/NI",
}

NUMERIC_FEATURES = [
    "Fase_num_t", "Idade_t", "Anos_na_PM_t", "Pedra_ordinal_t", "INDE_t",
    "IAA_t", "IEG_t", "IPS_t", "IPP_t", "IDA_t", "Nota_Mat_t", "Nota_Por_t",
    "IPV_t", "Defasagem_t", "IPP_disponivel_t",
]
CATEGORICAL_FEATURES = ["Genero_t", "Instituicao_grp_t", "Ano_t"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

COLUNAS_CSV_OBRIGATORIAS = [
    "Idade", "Fase", "Pedra", "Genero", "Instituicao", "Ano_ingresso", "Ano",
    "INDE", "IAA", "IEG", "IPS", "IDA", "Nota_Mat", "Nota_Por", "IPV", "Defasagem",
]


# ----------------------------------------------------------------------------
# Funções auxiliares — replicam exatamente a limpeza/feature engineering
# feitas no notebook, para que o app aceite dados "crus" (ex.: em um CSV
# exportado da planilha do PEDE) e não apenas valores já codificados.
# ----------------------------------------------------------------------------

def extract_fase_num(s):
    """Extrai o número da fase a partir de textos como 'FASE 3', '3P', 'ALFA'."""
    s = str(s).upper().strip()
    if s.startswith("ALFA"):
        return 0
    m = re.search(r"\d+", s)
    return int(m.group()) if m else np.nan


def group_instituicao(s):
    """Agrupa o texto livre de instituição de ensino em 4 categorias, igual ao notebook."""
    if pd.isna(s):
        return "Outra/NI"
    s_low = str(s).lower()
    if "públic" in s_low or "publica" in s_low:
        return "Pública"
    if "privada" in s_low:
        return "Privada"
    if "concluiu" in s_low or "universit" in s_low:
        return "Concluiu/Universitário"
    return "Outra/NI"


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def obter_importancias(modelo, top_n=10):
    """Extrai a importância de cada atributo diretamente do pipeline carregado."""
    try:
        prep = modelo.named_steps["prep"]
        clf = modelo.named_steps["clf"]
        nomes = prep.get_feature_names_out()
        if hasattr(clf, "feature_importances_"):
            valores = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            valores = np.abs(np.ravel(clf.coef_))
        else:
            return None
        imp = pd.Series(valores, index=nomes).sort_values(ascending=False).head(top_n)
        return imp.rename("Importância")
    except Exception:
        return None


def preparar_lote(df_raw: pd.DataFrame):
    """Converte um CSV com colunas 'cruas' (Idade, Fase, Pedra, ...) no formato
    exato que o pipeline espera (as mesmas colunas '_t' usadas no treino)."""
    faltantes = [c for c in COLUNAS_CSV_OBRIGATORIAS if c not in df_raw.columns]
    if faltantes:
        raise ValueError(
            "Colunas obrigatórias ausentes no arquivo: " + ", ".join(faltantes)
        )

    df = df_raw.copy()
    saida = pd.DataFrame(index=df.index)

    saida["Fase_num_t"] = df["Fase"].apply(extract_fase_num)
    saida["Idade_t"] = pd.to_numeric(df["Idade"], errors="coerce")
    saida["Anos_na_PM_t"] = (
        pd.to_numeric(df["Ano"], errors="coerce")
        - pd.to_numeric(df["Ano_ingresso"], errors="coerce")
    )

    pedra_norm = df["Pedra"].astype(str).str.strip().replace({"Agata": "Ágata"})
    saida["Pedra_ordinal_t"] = pedra_norm.map(PEDRA_MAP)

    diretos = [
        ("INDE_t", "INDE"), ("IAA_t", "IAA"), ("IEG_t", "IEG"), ("IPS_t", "IPS"),
        ("IDA_t", "IDA"), ("Nota_Mat_t", "Nota_Mat"), ("Nota_Por_t", "Nota_Por"),
        ("IPV_t", "IPV"), ("Defasagem_t", "Defasagem"),
    ]
    for col_saida, col_raw in diretos:
        saida[col_saida] = pd.to_numeric(df[col_raw], errors="coerce")

    if "IPP" in df.columns:
        ipp_num = pd.to_numeric(df["IPP"], errors="coerce")
        saida["IPP_t"] = ipp_num
        saida["IPP_disponivel_t"] = ipp_num.notnull().astype(int)
    else:
        saida["IPP_t"] = np.nan
        saida["IPP_disponivel_t"] = 0

    saida["Genero_t"] = df["Genero"].astype(str).replace(
        {"Menina": "Feminino", "Menino": "Masculino"}
    )
    saida["Instituicao_grp_t"] = df["Instituicao"].apply(group_instituicao)
    saida["Ano_t"] = df["Ano"].astype(str)

    meta = pd.DataFrame(index=df.index)
    for col in ["RA", "Nome"]:
        if col in df.columns:
            meta[col] = df[col]
    meta["Fase"] = df["Fase"]
    meta["Pedra"] = df["Pedra"]
    meta["Defasagem_atual"] = df["Defasagem"]

    return saida[ALL_FEATURES], meta


def gerar_csv_modelo() -> bytes:
    exemplo = pd.DataFrame([{
        "RA": "aluno_001", "Nome": "Exemplo", "Idade": 12, "Fase": "FASE 3",
        "Pedra": "Ágata", "Genero": "Feminino", "Instituicao": "Escola Pública",
        "Ano_ingresso": 2022, "Ano": 2024, "INDE": 6.8, "IAA": 7.0, "IEG": 7.5,
        "IPS": 6.5, "IPP": 7.2, "IDA": 6.0, "Nota_Mat": 6.5, "Nota_Por": 6.0,
        "IPV": 6.2, "Defasagem": -1,
    }])
    return exemplo.to_csv(index=False).encode("utf-8")


def classificar_faixa(p: float) -> str:
    if p >= 0.66:
        return "🔴 Alto"
    if p >= 0.33:
        return "🟡 Moderado"
    return "🟢 Baixo"


# ----------------------------------------------------------------------------
# Carregamento do modelo
# ----------------------------------------------------------------------------

st.title("🎯 Passos Mágicos — Risco de Defasagem Escolar")
st.caption(
    "Estima a probabilidade de um aluno estar com defasagem (fase abaixo do "
    "ideal) no próximo ciclo, a partir dos indicadores do PEDE no ano atual."
)

model = load_model(MODEL_PATH)
if model is None:
    st.error(
        f"Não encontrei o arquivo do modelo em `{MODEL_PATH}`. Coloque o "
        f"arquivo salvo com `joblib.dump(pipe_final, '{MODEL_PATH}')` na "
        f"mesma pasta deste `app.py` e recarregue a página."
    )
    st.stop()

tab_individual, tab_lote, tab_sobre = st.tabs(
    ["📋 Avaliação individual", "📁 Avaliação em lote (CSV)", "ℹ️ Sobre o modelo"]
)

# ----------------------------------------------------------------------------
# TAB 1 — Avaliação individual
# ----------------------------------------------------------------------------

with tab_individual:
    st.subheader("Dados do aluno (ano corrente)")

    with st.form("form_individual"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Dados gerais**")
            idade = st.number_input("Idade", min_value=5, max_value=25, value=12)
            genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
            instituicao_label = st.selectbox(
                "Instituição de ensino", list(INSTITUICAO_MAP.keys())
            )
            ano_ingresso = st.number_input(
                "Ano de ingresso na Passos Mágicos",
                min_value=2010, max_value=2030, value=2022,
            )
            ano_ref = st.number_input(
                "Ano de referência desta avaliação",
                min_value=2022, max_value=2030, value=2025,
                help=(
                    "Ano dos indicadores informados abaixo (o modelo prevê o "
                    "risco para o ano seguinte a este)."
                ),
            )

        with col2:
            st.markdown("**Histórico escolar**")
            fase_label = st.selectbox("Fase atual", list(FASE_MAP.keys()))
            pedra = st.selectbox("Pedra atual", list(PEDRA_MAP.keys()))
            defasagem = st.number_input(
                "Defasagem atual (fase atual − fase ideal)",
                min_value=-8, max_value=5, value=0,
                help="Negativo = aluno atrás do esperado; 0 ou positivo = no nível ideal ou à frente.",
            )
            nota_mat = st.slider("Nota de Matemática", 0.0, 10.0, 6.0, 0.1)
            nota_por = st.slider("Nota de Português", 0.0, 10.0, 6.0, 0.1)

        with col3:
            st.markdown("**Indicadores PEDE (0 a 10)**")
            inde = st.slider("INDE", 0.0, 10.0, 6.5, 0.1, help="Índice do Desenvolvimento Educacional — nota-síntese do aluno.")
            iaa = st.slider("IAA", 0.0, 10.0, 7.0, 0.1, help="Indicador de Autoavaliação.")
            ieg = st.slider("IEG", 0.0, 10.0, 7.0, 0.1, help="Indicador de Engajamento.")
            ips = st.slider("IPS", 0.0, 10.0, 6.5, 0.1, help="Indicador Psicossocial.")
            ida = st.slider("IDA", 0.0, 10.0, 6.0, 0.1, help="Indicador de Desempenho Acadêmico.")
            ipv = st.slider("IPV", 0.0, 10.0, 6.0, 0.1, help="Indicador do Ponto de Virada.")
            ipp_disponivel = st.checkbox("Avaliação psicopedagógica (IPP) disponível?", value=True)
            ipp = st.slider("IPP", 0.0, 10.0, 7.0, 0.1, disabled=not ipp_disponivel,
                             help="Indicador Psicopedagógico. Só existe a partir de 2023; se o aluno não tem essa avaliação, desmarque a caixa acima.")

        enviado = st.form_submit_button("Calcular probabilidade de risco", type="primary")

    if enviado:
        anos_na_pm = int(ano_ref) - int(ano_ingresso)
        if anos_na_pm < 0:
            st.warning("O ano de ingresso é posterior ao ano de referência — confira os valores.")

        linha = pd.DataFrame([{
            "Fase_num_t": FASE_MAP[fase_label],
            "Idade_t": idade,
            "Anos_na_PM_t": anos_na_pm,
            "Pedra_ordinal_t": PEDRA_MAP[pedra],
            "INDE_t": inde,
            "IAA_t": iaa,
            "IEG_t": ieg,
            "IPS_t": ips,
            "IPP_t": ipp if ipp_disponivel else np.nan,
            "IDA_t": ida,
            "Nota_Mat_t": nota_mat,
            "Nota_Por_t": nota_por,
            "IPV_t": ipv,
            "Defasagem_t": defasagem,
            "IPP_disponivel_t": int(ipp_disponivel),
            "Genero_t": genero,
            "Instituicao_grp_t": INSTITUICAO_MAP[instituicao_label],
            "Ano_t": str(int(ano_ref)),
        }])

        try:
            proba = float(model.predict_proba(linha[ALL_FEATURES])[:, 1][0])
            st.session_state["resultado_individual"] = proba
        except Exception as e:
            st.error(f"Erro ao calcular a previsão: {e}")

    if "resultado_individual" in st.session_state:
        proba = st.session_state["resultado_individual"]
        st.markdown("---")
        st.subheader("Resultado")

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Probabilidade de defasagem no próximo ciclo", f"{proba:.1%}")
            st.write(f"Nível de risco: **{classificar_faixa(proba)}**")
        with col_b:
            st.progress(min(max(proba, 0.0), 1.0))
            limiar = st.slider(
                "Limiar de decisão para sinalizar o aluno como 'em risco'",
                0.0, 1.0, 0.5, 0.01, key="limiar_individual",
                help=(
                    "Um limiar mais baixo sinaliza mais alunos (maior recall, "
                    "mais falsos positivos); um limiar mais alto sinaliza só "
                    "os casos mais claros. Ajuste conforme a capacidade da "
                    "equipe de atender os alunos sinalizados."
                ),
            )
            if proba >= limiar:
                st.error(f"⚠️ Aluno sinalizado como **em risco** de defasagem no próximo ciclo.")
            else:
                st.success("✅ Aluno sem sinalização de risco no limiar atual.")

        imp = obter_importancias(model)
        if imp is not None:
            with st.expander("Quais fatores mais pesaram nessa previsão (modelo geral)"):
                st.bar_chart(imp)
                st.caption(
                    "Importância dos atributos calculada a partir do modelo carregado "
                    "— reflete o que mais pesa nas previsões em geral, não apenas neste aluno."
                )

# ----------------------------------------------------------------------------
# TAB 2 — Avaliação em lote (CSV)
# ----------------------------------------------------------------------------

with tab_lote:
    st.subheader("Avaliar vários alunos de uma vez")
    st.write(
        "Envie uma planilha CSV com os dados atuais de vários alunos para "
        "gerar uma lista priorizada, do maior para o menor risco — o mesmo "
        "uso feito no notebook para a coorte de alunos ativos."
    )

    st.download_button(
        "⬇️ Baixar modelo de planilha (CSV)",
        data=gerar_csv_modelo(),
        file_name="modelo_dados_alunos.csv",
        mime="text/csv",
    )
    st.caption(
        "Colunas obrigatórias: " + ", ".join(COLUNAS_CSV_OBRIGATORIAS) +
        ". Colunas opcionais: RA, Nome, IPP."
    )

    arquivo = st.file_uploader("Arquivo CSV", type=["csv"])

    if arquivo is not None:
        try:
            df_raw = pd.read_csv(arquivo)
            st.write(f"{len(df_raw)} registros encontrados. Pré-visualização:")
            st.dataframe(df_raw.head())

            if st.button("Calcular risco para todos os alunos"):
                X_lote, meta = preparar_lote(df_raw)
                probas = model.predict_proba(X_lote[ALL_FEATURES])[:, 1]
                resultado = meta.copy()
                resultado["Probabilidade_Risco"] = probas
                resultado["Nivel_Risco"] = resultado["Probabilidade_Risco"].apply(classificar_faixa)
                resultado = resultado.sort_values("Probabilidade_Risco", ascending=False).reset_index(drop=True)
                st.session_state["resultado_lote"] = resultado
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

    if "resultado_lote" in st.session_state:
        resultado = st.session_state["resultado_lote"]
        st.markdown("---")
        st.subheader("Ranking de risco")

        limiar_lote = st.slider(
            "Limiar de prioridade", 0.0, 1.0, 0.5, 0.01, key="limiar_lote",
            help="Alunos com probabilidade acima deste valor são contados como prioritários abaixo.",
        )
        n_prioridade = int((resultado["Probabilidade_Risco"] >= limiar_lote).sum())
        st.write(
            f"**{n_prioridade}** de **{len(resultado)}** alunos sinalizados para "
            f"atenção prioritária (probabilidade ≥ {limiar_lote:.0%})."
        )

        st.dataframe(resultado, use_container_width=True)

        csv_saida = resultado.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar ranking completo (CSV)",
            data=csv_saida,
            file_name="ranking_risco_defasagem.csv",
            mime="text/csv",
        )

        bins = pd.cut(resultado["Probabilidade_Risco"], bins=np.linspace(0, 1, 11), include_lowest=True)
        contagem = bins.value_counts().sort_index()
        contagem.index = [str(i) for i in contagem.index]
        st.caption("Distribuição das probabilidades previstas na base enviada:")
        st.bar_chart(contagem)

# ----------------------------------------------------------------------------
# TAB 3 — Sobre o modelo
# ----------------------------------------------------------------------------

with tab_sobre:
    st.subheader("Como o modelo funciona")
    st.markdown(
        """
- **Pergunta que o modelo responde:** dados os indicadores do aluno no ano
  corrente (*t*), qual a probabilidade de ele estar com **defasagem**
  (fase abaixo da ideal) no ano seguinte (*t+1*)?
- **Como foi treinado:** com as transições reais observadas entre
  2022→2023 e 2023→2024, usando **apenas indicadores do ano anterior** para
  prever o ano seguinte — isso evita que o modelo "veja" informação do
  próprio ano que está tentando prever.
- **Separação treino/teste:** feita por aluno (`GroupShuffleSplit`), para
  que nenhum aluno apareça ao mesmo tempo em treino e teste.
- **Modelo:** Random Forest (com pré-processamento de imputação de valores
  faltantes, padronização de numéricas e one-hot de categóricas dentro do
  mesmo pipeline). Desempenho no conjunto de teste: ROC-AUC ≈ 0,90.
- **Atributos mais relevantes:** a defasagem atual e o Ponto de Virada
  (IPV) são os sinais mais fortes, seguidos pela fase atual e pelo INDE —
  ou seja, defasagem tende a persistir se nada muda.
        """
    )

    st.subheader("Sobre o campo 'Ano de referência'")
    st.markdown(
        """
O modelo só aprendeu padrões específicos para os anos **2022** e **2023**
como "ano corrente" (as duas transições disponíveis nos dados). Para
qualquer outro ano informado (ex.: 2024, 2025), esse campo não adiciona
nem atrapalha a previsão — o modelo simplesmente ignora esse valor
específico e usa só os demais indicadores. Ou seja, é seguro usar o ano
mais recente disponível.
        """
    )

    st.subheader("Limitações e uso recomendado")
    st.markdown(
        """
- Baseado em apenas duas transições anuais observadas (2022→2023 e
  2023→2024); o modelo deve ser **re-treinado a cada novo ciclo do PEDE**
  para ganhar robustez.
- O IPP só existe a partir de 2023 — para alunos sem essa avaliação,
  desmarque a caixa correspondente (o modelo trata a ausência de forma
  adequada).
- Alunos muito novos no programa, sem INDE consolidado ainda, devem ser
  reavaliados assim que a primeira avaliação estiver disponível.
- Esta é uma ferramenta de **priorização de apoio pedagógico e
  psicossocial** — não deve ser usada para decisões de exclusão de alunos,
  e seu desempenho por gênero e tipo de instituição deve ser monitorado
  ao longo do tempo para garantir equidade.
        """
    )

st.markdown("---")
st.caption(
    "Aplicação gerada a partir do modelo treinado em "
    "`passos_magicos_analise_completa.ipynb` (Pergunta 9 — modelo preditivo de risco)."
)
