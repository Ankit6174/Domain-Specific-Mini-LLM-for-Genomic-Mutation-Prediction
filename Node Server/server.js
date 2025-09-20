const express = require('express');
const path = require('path');
const dotenv = require('dotenv');
const axios = require('axios');
const DB = require("./config/contectDB");
const ContectSchema = require('./models/contect');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 8001;

const data = [
    {
        title: 'Melting Temperature',
        content: '45°C',
        percentage: 10,
        para: 'From Average'
    },
    {
        title: 'Molecular Weight',
        content: '660g/mol',
        percentage: 60,
        para: 'From Average'
    },
    {
        title: 'Genomic Posistion',
        content: '40324',
        percentage: 78,
        para: 'Location Confidence'
    },
    {
        title: 'Chromosome',
        content: '12',
        percentage: 'High',
        para: 'Gene Density'
    },
    {
        title: 'Muatation Type',
        content: 'A → T',
        percentage: 82,
        para: 'Effect Confidence'
    }
];

const label = [
    {
        'Pathogenic': 82
    },
    [
        'Benign',
        'Likely Benign',
        'Uncertain Significance',
        'Likely Pathogenic',
        'CIP'
    ],
    [
        90,
        55,
        70,
        30,
        15
    ]
];

const slicingGC = [
    {
        'max': 82
    },
    [
        50, 
        40, 
        60, 
        55, 
        82, 
        45
    ]
]

const nucleoFreq = [
    ['A', 60],
    ['T', 50],
    ['G', 90],
    ['C', 80]
]

const at_gc_content = [
    ['AT', 45, '#2a2a2a'],
    ['GC', 55, 'rgba(255, 0, 0, 0.48)'],
]

const getComplement = ((nucleo) => ({ G: 'C', C: 'G', A: 'T', T: 'A' }[nucleo]));

const dna = 'GGTGGCCGCTGTGGCCTGTGCCCAAGTGCCTAAGATAACCCTCATCATTGGGGGCTCCTATGGAGCCGGAAACTATGGGATGTGTGGCAGAGCATATAGGTAGGTGTCATGATTTTCTCTGAAACAAAGAAACATGCTTCAAGTATAAAATACATGGTCAGTTTATTTCAGGTGTATTTGAAATATAGAATGCCATTCCCA';

const mRNA = [
    [dna[0], getComplement(dna[0])],
    [],
    [dna.slice(-1), getComplement(dna.slice(-1))]
];

for (nucleo of dna.slice(1, -1)) {
    mRNA[1].push([nucleo, getComplement(nucleo)])
}

// Configuration
app.set('view engine', 'ejs');

// Middlewares
app.use(express.urlencoded({extended: true}));
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.render("Home");
});

app.get('/prediction', (req, res) => {
    res.render("Prediction", {data, label, slicingGC, nucleoFreq, at_gc_content, mRNA});
});

app.get('/prediction-form', (req, res) => {
    res.render("Prediction_Form");
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
    // let { dnasequence, reference, alternate, mutation, chromosome, genomicPosition } = req.body;
    const data = req.body;
    console.log("Received data for prediction:", data);

    axios.post("http://127.0.0.1:7000/predict", data)
        .then((responce) => {
            let prediction = responce.data;
            console.log(prediction)
            res.send(prediction);
        })
        .catch((error) => {
            console.log(error);
            res.sendStatus(512);

        });
});

app.listen(PORT, () => {
    console.log(`Server Running On http://localhost:${PORT}/`);
});