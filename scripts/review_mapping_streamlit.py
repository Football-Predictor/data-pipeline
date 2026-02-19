#!/usr/bin/env python3
"""Streamlit UI for reviewing player_map_review.csv with fuzzy candidate picker

Run with:
  streamlit run scripts/review_mapping_streamlit.py

Features:
- Browse `player_map_review.csv` rows (status==review or unmatched)
- Show top probable FIFA candidates (token-blocked + rapidfuzz scoring)
- Allow selecting a candidate, accepting mapping, or entering a manual FIFA id/attributes
- When accepting, update `data/mappings/player_map.csv` and `player_map_review.csv`
- Optionally append manual FIFA rows to `data/cache/fifa_players_1000.parquet` and copy attributes into `players_train_1000.parquet`

Note: this tool writes directly to project CSV / parquet files.
"""
# /// script
# dependencies = ["pandas","pyarrow","rapidfuzz","streamlit"]
# ///
import streamlit as st
from pathlib import Path
import pandas as pd
import unicodedata
try:
    from rapidfuzz import process, fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    process = None
    fuzz = None
    _HAS_RAPIDFUZZ = False

# resolve project root relative to this script so Streamlit can be launched from any cwd
ROOT = Path(__file__).resolve().parents[1] / ''
ROOT = ROOT.parent if ROOT.name == 'scripts' else Path(__file__).resolve().parents[1]
ROOT = ROOT / 'data'
REVIEW_P = ROOT / 'mappings' / 'player_map_review.csv'
ACCEPT_P = ROOT / 'mappings' / 'player_map.csv'
FIFA_P = ROOT / 'cache' / 'fifa_players_1000.parquet'
PLAYERS_P = ROOT / 'processed' / 'players_train_1000.parquet'

ATTRS = ['pace','shooting','passing','dribbling','defending','physic']

st.set_page_config(page_title='Player mapping reviewer', layout='wide')

if not _HAS_RAPIDFUZZ:
    st.warning("`rapidfuzz` is not installed in this runtime — fuzzy candidate lookup is disabled. "
               "Install it with `uv add rapidfuzz` or launch Streamlit from the project venv:\n"
               "& \".venv\\Scripts\\python.exe\" -m streamlit run scripts/review_mapping_streamlit.py")

@st.cache_data
def load_review():
    # be tolerant if the review CSV is missing (create empty frame with expected cols)
    if not REVIEW_P.exists():
        cols = ['player_id_sb','player_name_sb','candidate_fifa_id','candidate_name','score','status','method']
        return pd.DataFrame(columns=cols)
    return pd.read_csv(REVIEW_P, dtype={'player_id_sb': object})

@st.cache_data
def load_fifa():
    df = pd.read_parquet(FIFA_P)
    # ensure sofifa_id exists
    if 'sofifa_id' not in df.columns:
        df = df.reset_index(drop=True)
        df['sofifa_id'] = (df.index + 1).astype(int)
    df['_norm_short'] = df.get('short_name','').fillna('').astype(str).apply(normalize_name)
    df['_norm_long'] = df.get('long_name','').fillna('').astype(str).apply(normalize_name)
    df['_norm_combined'] = (df['_norm_short'] + ' || ' + df['_norm_long']).fillna('')
    return df

@st.cache_data
def load_players():
    return pd.read_parquet(PLAYERS_P)


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join([c for c in s if not unicodedata.combining(c)])
    s = s.replace("'", "").replace('"','').replace('.','')
    s = ' '.join(s.split())
    return s


def token_block_candidates(sb_name, fifa_df, max_candidates=5000):
    sbn = normalize_name(sb_name)
    toks = [t for t in sbn.split() if len(t)>1]
    if not toks:
        return fifa_df
    last = toks[-1]
    cand = fifa_df[fifa_df['_norm_combined'].str.contains(last, na=False)]
    if len(cand) > max_candidates:
        cand = cand.sample(max_candidates, random_state=1)
    return cand


def get_top_candidates(sb_name, fifa_df, top_n=10):
    # If rapidfuzz is missing in this runtime (for example when using `uv tool run`),
    # return an empty candidate set and rely on the UI warning shown at startup.
    if not _HAS_RAPIDFUZZ:
        return []
    cand_df = token_block_candidates(sb_name, fifa_df, max_candidates=5000)
    if cand_df.empty:
        return []
    choices = cand_df['_norm_combined'].tolist()
    res = process.extract(normalize_name(sb_name), choices, scorer=fuzz.token_set_ratio, limit=top_n)
    out = []
    for match_str, score, idx in res:
        row = cand_df.iloc[idx]
        out.append({
            'sofifa_id': row.get('sofifa_id'),
            'short_name': row.get('short_name'),
            'long_name': row.get('long_name'),
            'player_positions': row.get('player_positions'),
            'score': int(score),
            **{a: row.get(a) for a in ATTRS}
        })
    return out


def append_accept_mapping(player_id_sb, player_name_sb, fifa_id, score, method='manual'):
    # append to data/mappings/player_map.csv
    row = {'player_id_sb': int(player_id_sb), 'player_name_sb': player_name_sb, 'fifa_id': int(fifa_id), 'score': float(score), 'method': method}
    if ACCEPT_P.exists():
        df = pd.read_csv(ACCEPT_P, dtype={'player_id_sb': int, 'fifa_id': object})
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(ACCEPT_P, index=False)


def update_review_status(idx, new_status, candidate_fifa_id=None, candidate_name=None, score=None):
    review = pd.read_csv(REVIEW_P, dtype={'player_id_sb': object})
    if idx not in review.index:
        # try by player_id_sb match
        pass
    review.at[idx,'status'] = new_status
    if candidate_fifa_id is not None:
        review.at[idx,'candidate_fifa_id'] = candidate_fifa_id
    if candidate_name is not None:
        review.at[idx,'candidate_name'] = candidate_name
    if score is not None:
        review.at[idx,'score'] = score
    review.to_csv(REVIEW_P, index=False)


def append_fifa_row_and_get_id(short_name, long_name, positions, attrs:dict):
    # append row to fifa_players_1000.parquet and return sofifa_id
    df = pd.read_parquet(FIFA_P)
    df = df.reset_index(drop=True)
    next_id = int(df['sofifa_id'].max())+1 if 'sofifa_id' in df.columns else len(df)+1
    new = {'sofifa_id': next_id, 'short_name': short_name, 'long_name': long_name, 'player_positions': positions}
    for a in ATTRS:
        new[a] = attrs.get(a)
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_parquet(FIFA_P, index=False)
    return next_id


def copy_fifa_attrs_into_players(player_name_sb, sofifa_id):
    # find fifa row and copy ATTRS into players_train_1000.parquet where player_name_sb matches
    players = pd.read_parquet(PLAYERS_P)
    fifa = pd.read_parquet(FIFA_P)
    if 'sofifa_id' in fifa.columns:
        row = fifa[fifa['sofifa_id']==int(sofifa_id)]
    else:
        row = fifa[fifa.index==int(sofifa_id)]
    if len(row)==0:
        return 0
    row = row.iloc[0]
    idxs = players[players['player_name_sb']==player_name_sb].index.tolist()
    for i in idxs:
        for a in ATTRS:
            if pd.isna(players.at[i,a]) and a in row.index and pd.notna(row.get(a)):
                players.at[i,a] = row.get(a)
    players.to_parquet(PLAYERS_P, index=False)
    return len(idxs)


# --- Streamlit UI ---

st.title('Mapping reviewer (Streamlit)')
review_df = load_review()
fifa_df = load_fifa()
players_df = load_players()

# filter rows for review/unmatched
filter_status = st.sidebar.selectbox('Filter rows by status', ('review','unmatched','all'))
if filter_status != 'all':
    candidates_df = review_df[review_df['status']==filter_status].reset_index(drop=True)
else:
    candidates_df = review_df.copy().reset_index(drop=True)

if candidates_df.empty:
    st.info('No rows matching filter')
    st.stop()

# session-backed row index so reviewer can auto-advance after actions
if 'row_index' not in st.session_state:
    st.session_state['row_index'] = 0
# reset row index when the filter changes
if st.session_state.get('last_filter_status') != filter_status:
    st.session_state['row_index'] = 0
    st.session_state['last_filter_status'] = filter_status
# clamp to valid range
st.session_state['row_index'] = min(st.session_state['row_index'], max(0, len(candidates_df) - 1))

row_index = st.sidebar.number_input('Row index (in filtered set)', min_value=0, max_value=len(candidates_df)-1, value=st.session_state['row_index'], step=1, key='row_index_input')
# keep session state in sync when user edits the widget
if int(row_index) != st.session_state['row_index']:
    st.session_state['row_index'] = int(row_index)

row = candidates_df.iloc[int(st.session_state['row_index'])]

st.subheader('StatsBomb player')
st.write('player_id_sb:', row.get('player_id_sb'))
st.write('player_name_sb:', row.get('player_name_sb'))
st.write('current status:', row.get('status'))
st.write('current candidate:', row.get('candidate_name'), row.get('candidate_fifa_id'), 'score=', row.get('score'))

st.markdown('---')

col1, col2 = st.columns([2,3])
with col1:
    st.markdown('### Most probable FIFA candidates')
    pb = get_top_candidates(row.get('player_name_sb'), fifa_df, top_n=12)
    if not pb:
        st.info('No candidates found by fuzzy search')
    else:
        options = [f"{p['sofifa_id']} — {p['short_name']} ({p['player_positions']}) — score={p['score']}" for p in pb]
        sel = st.selectbox('Choose candidate to accept', options)
        sel_idx = options.index(sel) if sel in options else 0
        chosen = pb[sel_idx]
        st.write('Attributes:')
        st.write({a: chosen.get(a) for a in ATTRS})
        if st.button('Accept selected candidate'):
            sofifa = chosen['sofifa_id']
            append_accept_mapping(row.get('player_id_sb'), row.get('player_name_sb'), sofifa, chosen['score'], method='streamlit_manual')
            update_review_status(int(row.name), 'accepted_manual', candidate_fifa_id=sofifa, candidate_name=chosen['short_name'], score=chosen['score'])
            copy_fifa_attrs_into_players(row.get('player_name_sb'), sofifa)
            st.success('Accepted mapping and copied attributes (if any).')
            # auto-advance to next row in the filtered set
            st.session_state['row_index'] = min(st.session_state.get('row_index', 0) + 1, len(candidates_df) - 1)
            st.experimental_rerun()

with col2:
    st.markdown('### Manual mapping / Add FIFA row')
    with st.form('manual_form'):
        manual_fifa_id = st.text_input('FIFA id (leave blank to create new FIFA row)')
        manual_short = st.text_input('FIFA short name', value=row.get('player_name_sb'))
        manual_long = st.text_input('FIFA long name', value='')
        manual_pos = st.text_input('player_positions (e.g. ST, CM)', value='')
        st.markdown('Attributes (optional, leave blank to skip)')
        manual_attrs = {}
        for a in ATTRS:
            v = st.text_input(a, value='')
            manual_attrs[a] = (float(v) if v.strip()!='' else None)
        submitted = st.form_submit_button('Add / accept manual mapping')
        if submitted:
            if manual_fifa_id.strip()=='':
                # create new fifa row and get id
                new_id = append_fifa_row_and_get_id(manual_short or row.get('player_name_sb'), manual_long, manual_pos, manual_attrs)
                append_accept_mapping(row.get('player_id_sb'), row.get('player_name_sb'), new_id, 100.0, method='streamlit_manual_added')
                update_review_status(int(row.name), 'accepted_manual', candidate_fifa_id=new_id, candidate_name=manual_short, score=100.0)
                copy_fifa_attrs_into_players(row.get('player_name_sb'), new_id)
                st.success(f'Added new FIFA row (sofifa_id={new_id}) and accepted mapping.')
                # auto-advance
                st.session_state['row_index'] = min(st.session_state.get('row_index', 0) + 1, len(candidates_df) - 1)
                st.experimental_rerun()
            else:
                try:
                    fid_int = int(manual_fifa_id)
                    append_accept_mapping(row.get('player_id_sb'), row.get('player_name_sb'), fid_int, 100.0, method='streamlit_manual')
                    update_review_status(int(row.name), 'accepted_manual', candidate_fifa_id=fid_int, candidate_name=manual_short or row.get('player_name_sb'), score=100.0)
                    copy_fifa_attrs_into_players(row.get('player_name_sb'), fid_int)
                    st.success(f'Accepted manual mapping to fifa_id={fid_int}.')
                    # auto-advance
                    st.session_state['row_index'] = min(st.session_state.get('row_index', 0) + 1, len(candidates_df) - 1)
                    st.experimental_rerun()
                except Exception as e:
                    st.error('Invalid FIFA id')

st.markdown('---')

if st.button('Mark as unmatched'):
    update_review_status(int(row.name), 'unmatched')
    st.success('Marked as unmatched')
    # auto-advance
    st.session_state['row_index'] = min(st.session_state.get('row_index', 0) + 1, len(candidates_df) - 1)
    st.experimental_rerun()

st.markdown('### Quick search across FIFA (name)')
q = st.text_input('Search FIFA names (substring)')
if q.strip()!='':
    qn = normalize_name(q)
    hits = fifa_df[fifa_df['_norm_combined'].str.contains(qn, na=False)].head(50)
    st.dataframe(hits[['sofifa_id','short_name','long_name','player_positions'] + [c for c in ATTRS if c in hits.columns]])

st.markdown('---')
if st.button('Reload data'):
    st.experimental_rerun()

st.caption('Edits persist to files in data/mappings/ and data/cache/. Use the reviewer to keep mapping quality high.')
