mkdir -p ~/.streamlit/
echo "[theme]
primaryColor='#477569'
backgroundColor='#fff700'
secondaryBackgroundColor='#171777'
textColor='#000006'
font = 'sans serif'
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
