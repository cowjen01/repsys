import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';

if (process.env.NODE_ENV === 'development') {
  // eslint-disable-next-line global-require
  const { worker } = require('./mocks/browser');
  worker.start({ onUnhandledRequest: 'bypass' });
}

const app = React.createElement(App);

ReactDOM.render(app, document.getElementById('root'));
