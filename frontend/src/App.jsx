import React from 'react';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Provider } from 'react-redux';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

import store from './store';
import Studio from './app/Studio';

const theme = createTheme({
  palette: {
    mode: 'dark',
  },
});

function App() {
  return (
    <Provider store={store}>
      <DndProvider backend={HTML5Backend}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <Studio />
        </ThemeProvider>
      </DndProvider>
    </Provider>
  );
}

export default App;
