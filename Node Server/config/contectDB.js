const mongoose = require('mongoose');

const contectDB = async() => {
    try {
        await mongoose.connect(process.env.MONGO_URL);
        console.log(`Connected -> ${process.env.MONGO_URL}`);
    } catch {
        console.log("Failed to connect to database");
    }
};

module.exports = contectDB;