import React, { useMemo } from 'react';
import pt from 'prop-types';
import { Provider, useSelector } from 'react-redux';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import red from '@mui/material/colors/red';

import store from './store';
import Studio from './app/Studio';
import { darkModeSelector } from './reducers/settings';

function ThemeWrapper({ children }) {
  const darkMode = useSelector(darkModeSelector);

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          ...(!darkMode
            ? {
                primary: {
                  main: '#212121',
                  light: '#484848',
                  dark: '#000000',
                },
                secondary: red,
                background: {
                  default: '#fafafa',
                },
              }
            : {
                mode: 'dark',
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

function App() {
  return (
    <Provider store={store}>
      <DndProvider backend={HTML5Backend}>
        <ThemeWrapper>
          <CssBaseline />
          <Studio />
        </ThemeWrapper>
      </DndProvider>
    </Provider>
  );
}

export default App;
