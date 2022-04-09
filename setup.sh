mkdir -p ~/.streamlit/
echo "[theme]
primaryColor='#394877'
backgroundColor='#440ebd'
secondaryBackgroundColor='#efa330'
textColor='#ffffff'
font = 'sans serif'
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
