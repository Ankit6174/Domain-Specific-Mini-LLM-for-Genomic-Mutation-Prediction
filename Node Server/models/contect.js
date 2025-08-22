const mongoose = require('mongoose');

const Schema = mongoose.Schema({
    name: {
        type: String,
        require: true
    
    },
    email: {
        type: String,
        require: true,
        unique: true
    },
    organization: {
        type: String,
    },
    mindTopic: {
        type: String,
        require: true
    },
    message: {
        type: String,
        require: true
    }
});

const ContectSchema = mongoose.model("contect", Schema);

module.exports = ContectSchema;