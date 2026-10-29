const MLModel = require('../models/MLModel');
const mlPredictionService = require('../services/mlPredictionService');

async function saveModel(req, res, next) {
    try {
        const { serializedData } = req.body;
        if (!serializedData) return res.status(400).json({ error: 'serializedData is required' });

        // Update existing or create new
        const model = await MLModel.findOneAndUpdate(
            { userId: req.userId, modelType: 'delay_predictor' },
            { serializedData, timestamp: new Date() },
            { upsert: true, new: true }
        );

        res.status(201).json({ message: 'Model saved successfully', timestamp: model.timestamp });
    } catch (err) {
        next(err);
    }
}

async function loadModel(req, res, next) {
    try {
        const model = await MLModel.findOne({ userId: req.userId, modelType: 'delay_predictor' });
        if (!model) return res.status(404).json({ error: 'No trained model found for this user' });

        res.json(model.serializedData);
    } catch (err) {
        next(err);
    }
}

async function predictSingle(req, res, next) {
    try {
        const vessel = req.body;
        if (!vessel) return res.status(400).json({ error: 'Vessel payload required' });
        const prediction = await mlPredictionService.predictVessel(vessel);
        res.json(prediction);
    } catch (err) {
        next(err);
    }
}

async function predictBatch(req, res, next) {
    try {
        const vessels = Array.isArray(req.body) ? req.body : (req.body.vessels || []);
        const predictions = await mlPredictionService.predictBatch(vessels);
        res.json({ count: predictions.length, predictions });
    } catch (err) {
        next(err);
    }
}

async function getMetrics(req, res, next) {
    try {
        const metrics = await mlPredictionService.getMetrics();
        res.json(metrics);
    } catch (err) {
        next(err);
    }
}

async function getConstraints(req, res, next) {
    try {
        const constraints = await mlPredictionService.getConstraints();
        res.json(constraints);
    } catch (err) {
        next(err);
    }
}

async function trainModel(req, res, next) {
    try {
        const { data } = req.body || {};
        const result = await mlPredictionService.trainModel(data);
        res.json(result);
    } catch (err) {
        next(err);
    }
}

async function healthCheck(req, res, next) {
    try {
        const health = await mlPredictionService.checkHealth();
        res.json(health);
    } catch (err) {
        next(err);
    }
}

module.exports = {
    saveModel,
    loadModel,
    predictSingle,
    predictBatch,
    getMetrics,
    getConstraints,
    trainModel,
    healthCheck,
};
