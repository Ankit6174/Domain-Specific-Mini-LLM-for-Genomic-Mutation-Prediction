const express = require('express');
const path = require('path');
const dotenv = require('dotenv');
const axios = require('axios');
const DB = require("./config/contectDB");
const ContectSchema = require('./models/contect');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 8001;

// Configuration
app.set('view engine', 'ejs');

// Middlewares
app.use(express.urlencoded({extended: true}));
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.render("Home");
});

app.get('/prediction-form', (req, res) => {
    res.render("Prediction_Form");
});

app.get('/test', (req, res) => {
    res.render("Test", { data, label, slicingGC, nucleoFreq, at_gc_content, comDNA });
});

DB();

app.post("/postContect", async (req, res) => {
    let { name, email, organization, mindTopic, message } = req.body;
    await ContectSchema.create({
        name, 
        email, 
        organization, 
        mindTopic, 
        message
    });
    res.sendStatus(200);
});

app.post("/predict", (req, res) => {
    const data = req.body;

    axios.post("https://ankitt6174-dna-mutation-prediction.hf.space/predict", data)
        .then((responce) => {
            let prediction = responce.data;
            
            res.render('Dashboard', {
                data: prediction.topData, 
                prediction_score: prediction.Prediction, 
                comDNA: prediction.comDNA,
                slicingGC: prediction.slicingGC,
                nucleoFreq: prediction.nucleoFreq,
                at_gc_content: prediction.at_gc_content,
                mutation_name: prediction.Mutation_Label,
                DNA: prediction.DNA,
                mRNA: prediction.mRNA,
                Protein: prediction.Protein
            });
        })
        .catch((error) => {
            console.log(error);
            res.sendStatus(512);

        });
});

app.listen(PORT, () => {
    console.log(`Server Running On http://localhost:${PORT}/`);
});