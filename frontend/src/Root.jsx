import React, { useMemo } from 'react';
import pt from 'prop-types';
import { Provider, useSelector } from 'react-redux';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import * as colors from '@mui/material/colors';
import { PersistGate } from 'redux-persist/integration/react';
import { BrowserRouter } from 'react-router-dom';

import { store, persistor } from './store';
import { darkModeSelector } from './reducers/settings';
import App from './components/App';

function ThemeWrapper({ children }) {
  const darkMode = useSelector(darkModeSelector);

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          ...(!darkMode
            ? {
                primary: {
                  main: '#121212',
                },
                secondary: colors.amber,
                background: {
                  default: '#fafafa',
                },
              }
            : {
                mode: 'dark',
                secondary: colors.amber,
              }),
        },
      }),
    [darkMode]
  );

  return <ThemeProvider theme={theme}>{children}</ThemeProvider>;
}

ThemeWrapper.propTypes = {
  children: pt.node.isRequired,
};

function Root() {
  return (
    <BrowserRouter>
      <Provider store={store}>
        <PersistGate loading={null} persistor={persistor}>
          <DndProvider backend={HTML5Backend}>
            <ThemeWrapper>
              <CssBaseline />
              <App />
            </ThemeWrapper>
          </DndProvider>
        </PersistGate>
      </Provider>
    </BrowserRouter>
  );
}

export default Root;
