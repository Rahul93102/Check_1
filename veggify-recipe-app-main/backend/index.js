const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Connect to MongoDB
const mongoURI = "mongodb+srv://mdalmamunit427:0eqHrVZeHcSFeCmr@veggify-recipe-app.da0qxhz.mongodb.net/veggify-recipe?retryWrites=true&w=majority&appName=veggify-recipe-app";

mongoose.connect(mongoURI).then(()=> console.log("Mongodb connected successfully!"));

// routes
const ItemRoutes = require("./src/routes/itemRoute");
const categoryRoutes = require("./src/routes/categoryRoute")

app.use('/api', ItemRoutes)
app.use('/api', categoryRoutes)

app.get('/', (req, res) => {
    res.send('Veggify Recipe App Server is running!')
  })


// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
