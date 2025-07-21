const express = require('express');
const path = require('path');

const app = express();
const PORT = 8000;

app.set('view engine', 'ejs');

app.use(express.urlencoded({extended: true}));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.render("Home");
});

app.listen(PORT, () => {
    console.log(`Server Running On http://localhost:${PORT}/`);
});