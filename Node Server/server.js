const express = require('express');
const path = require('path');

const app = express();
const PORT = 8000;

// Configuration
app.set('view engine', 'ejs');

// Middlewares
app.use(express.urlencoded({extended: true}));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.render("Home");
});

app.get('/prediction', (req, res) => {
    res.render("Prediction");
});

app.listen(PORT, () => {
    console.log(`Server Running On http://localhost:${PORT}/`);
});