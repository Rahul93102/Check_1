
const Item = require("../model/itemModel");

const getAllItems = async (req, res, next) =>{
    const result = await Item.find().sort({ createdAt: -1 });
    res.json(result);
};

const getSeachedItems = async (req, res) => {
    const { q } = req.query;
    try {
      let items;
      if (q) {
        items = await Item.find({ name: { $regex: q, $options: 'i' } });
      }
      res.json(items);
    } catch (error) {
      res.status(500).json({ message: 'No items found' });
    }
  }

  const getSingleItem = async (req, res) => {
    const { id } = req.params;
    try {
      const item = await Item.findById(id);
      res.json(item);
    } catch (error) {
      res.status(500).json({ message: 'Not found any item' });
    }
  }

  module.exports = {
    getAllItems,
    getSeachedItems,
    getSingleItem
  };