const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const { spawn } = require('child_process');
require('dotenv').config();

const app = express();

// Middleware
app.use((req, res, next) => {
    console.log(`[Server] ${req.method} ${req.path}`);
    next();
});
app.use(cors());
app.use(express.json());

// MongoDB
mongoose.connect(process.env.MONGODB_URI)
    .then(() => console.log('✅ MongoDB connected'))
    .catch(err => console.error('MongoDB connection error:', err.message));

// 🚀 Routes
app.use('/api/auth', require('./routes/authRoutes'));
app.use('/api/data', require('./routes/dataRoutes'));
app.use('/api/optimizations', require('./routes/optimizations'));
app.use('/api/dashboard', require('./routes/dashboardRoutes'));
app.use('/api/ml', require('./routes/mlRoutes'));
app.use('/api/simulation', require('./routes/simulationRoutes'));
app.use('/api/vessels', require('./routes/vesselRoutes'));
app.use('/api/budget', require('./routes/budgetRoutes'));
app.use('/api/inventory', require('./routes/inventoryRoutes'));

// 📁 Serve static files from the project root
app.use(express.static(path.resolve(__dirname, '..'), {
    etag: false,
    lastModified: false,
    setHeaders: (res) => {
        res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate');
        res.setHeader('Pragma', 'no-cache');
    }
}));

// 🔄 SPA Fallback: Serve index.html for non-API requests
app.use((req, res, next) => {
    if (req.path.startsWith('/api')) {
        return res.status(404).json({ error: `API route not found: ${req.path}` });
    }
    res.sendFile(path.resolve(__dirname, '..', 'index.html'));
});

// 🛠️ Global Error Handler
const errorHandler = require('./middleware/errorHandler');
app.use(errorHandler);

// Helper to spawn Python ML Prediction microservice
let pythonProcess = null;
function startPythonMLService() {
    if (process.env.AUTO_START_ML_SERVICE === 'false') return;

    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const scriptPath = path.join(__dirname, '..', 'ml', 'service.py');

    try {
        console.log(`[Server] Spawning Python ML Service on port 5001...`);
        pythonProcess = spawn(pythonCmd, [scriptPath, '--server', '--port', '5001'], {
            stdio: 'pipe',
            detached: false,
        });

        pythonProcess.stdout.on('data', (d) => console.log(`[Python ML] ${d.toString().trim()}`));
        pythonProcess.stderr.on('data', (d) => console.error(`[Python ML Err] ${d.toString().trim()}`));

        pythonProcess.on('error', (err) => {
            console.warn(`[Server] Could not spawn Python ML service: ${err.message}. Backend will use CLI/fallback mode.`);
        });

        process.on('exit', () => {
            if (pythonProcess) pythonProcess.kill();
        });
    } catch (e) {
        console.warn(`[Server] Python auto-spawn failed (${e.message}). Express will connect via CLI or fallback.`);
    }
}

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    startPythonMLService();
});
