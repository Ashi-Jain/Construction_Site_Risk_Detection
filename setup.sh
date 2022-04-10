mkdir -p ~/.streamlit/
echo "[theme]
primaryColor='#477569'
backgroundColor='#0b1077'
secondaryBackgroundColor='#ff9916'
textColor='#fefeff'
font = 'sans serif'
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
