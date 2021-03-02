import React, { useState, useEffect, useReducer } from 'react';
import './App.css';
import Form from './Form.js';
import ScraperWrapper from './Webscraper/ScraperWrapper';
import logo from './logoContext.png'
function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} alt="ContextCheck"></img>
      </header>
      <Form/>
      <ScraperWrapper />
    </div>
  );
}

export default App;
