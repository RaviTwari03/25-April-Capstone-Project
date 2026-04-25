# Retail Demand Forecasting Web Interface

A modern, responsive web interface for interacting with the Retail Demand Forecasting API.

## 🚀 Quick Start

### Prerequisites

1. **FastAPI Server Running**: Make sure the FastAPI server is running on `http://localhost:8000`
2. **Python 3.8+**: Required for the web server

### Option 1: Using the Python Web Server (Recommended)

1. **Start the FastAPI server:**
   ```bash
   cd "MLOps Fundamental project"
   python src/api/main.py
   ```

2. **Start the web server in a new terminal:**
   ```bash
   cd "MLOps Fundamental project"
   python web_server.py
   ```

3. **Open your browser:**
   - The web server will automatically open `http://localhost:3000`
   - If not, manually navigate to `http://localhost:3000/index.html`

### Option 2: Direct File Access

You can also open `index.html` directly in your browser, but you'll need to handle CORS issues by starting the FastAPI server with CORS enabled (which it already is).

## 🎯 Features

### 📊 Model Information Dashboard
- Real-time model status and performance metrics
- Model version information
- Training statistics (RMSE, MAE, R²)

### 🔮 Single Prediction
- Interactive form for single product demand prediction
- Real-time confidence interval visualization
- Input validation and error handling

### 📈 Batch Prediction
- Dynamic table for multiple predictions
- Sample data loading for testing
- Batch processing with performance metrics
- Exportable results

### 🚨 Drift Detection
- One-click drift detection trigger
- Background processing with status updates

### 🎨 Modern UI Features
- Responsive design for all devices
- Gradient backgrounds and smooth animations
- Real-time API status indicator
- Loading states and progress indicators
- Toast notifications for user feedback

## 🔧 Technical Details

### API Endpoints Used
- `GET /health` - API health check
- `GET /model/info` - Model information
- `POST /predict-demand` - Single prediction
- `POST /predict-demand/batch` - Batch prediction
- `POST /drift-detect` - Drift detection

### Frontend Technologies
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with gradients and animations
- **JavaScript (ES6+)** - Modern JavaScript with async/await
- **Bootstrap 5** - Responsive UI framework
- **Font Awesome** - Icon library

### Browser Compatibility
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## 📱 Mobile Support

The interface is fully responsive and works on:
- Mobile phones (iOS and Android)
- Tablets
- Desktop computers

## 🔍 Troubleshooting

### Common Issues

1. **"API Disconnected" Status**
   - Ensure FastAPI server is running on `http://localhost:8000`
   - Check for firewall issues
   - Verify both servers are on different ports

2. **CORS Errors**
   - The FastAPI server already has CORS enabled
   - If using direct file access, use the Python web server instead

3. **Prediction Failures**
   - Check all required fields are filled (Product ID, Date, Region)
   - Ensure date format is YYYY-MM-DD
   - Verify the model files exist in the `models/` directory

4. **Loading Issues**
   - Refresh the page
   - Check browser console for errors
   - Ensure all files are present in the `web/` directory

### Debug Mode

Open browser developer tools (F12) to see:
- API request/response logs
- JavaScript console errors
- Network activity

## 🎨 Customization

### Changing Colors
Edit `style.css` and modify the CSS variables at the top:
```css
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --success-gradient: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    /* ... */
}
```

### Adding New Regions
Edit both `index.html` and `script.js` to add new region options.

### Modifying API Endpoints
Update the `API_BASE_URL` constant in `script.js` if your API runs on a different port or domain.

## 📁 File Structure

```
web/
├── index.html          # Main HTML file
├── style.css           # Custom styling
├── script.js           # JavaScript functionality
└── README.md           # This file

web_server.py           # Python web server (in project root)
```

## 🚀 Production Deployment

For production deployment, consider:
1. **Static File Hosting**: Deploy to Netlify, Vercel, or GitHub Pages
2. **API Security**: Add authentication to the FastAPI endpoints
3. **HTTPS**: Use SSL certificates for secure connections
4. **CDN**: Use a CDN for faster asset delivery

## 📞 Support

If you encounter issues:
1. Check the FastAPI server logs
2. Verify all model files exist
3. Test API endpoints directly using curl or Postman
4. Check browser console for JavaScript errors

---

**Last Updated**: April 25, 2026  
**Version**: 1.0.0
