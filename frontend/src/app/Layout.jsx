import React, { useMemo, useState } from 'react';
import pt from 'prop-types';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import MuiAppBar from '@mui/material/AppBar';
import Typography from '@mui/material/Typography';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import { useDispatch, useSelector } from 'react-redux';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import * as colors from '@mui/material/colors';
import BubbleChartIcon from '@mui/icons-material/BubbleChart';
import Button from '@mui/material/Button';
import MenuItem from '@mui/material/MenuItem';
import Menu from '@mui/material/Menu';
import Grid from '@mui/material/Grid';
import DashboardIcon from '@mui/icons-material/Dashboard';
import { useHotkeys } from 'react-hotkeys-hook';
import DonutSmallIcon from '@mui/icons-material/DonutSmall';
import TableRowsIcon from '@mui/icons-material/TableRows';
import SettingsIcon from '@mui/icons-material/Settings';
import ListItemText from '@mui/material/ListItemText';
import ListItemIcon from '@mui/material/ListItemIcon';
import Divider from '@mui/material/Divider';

import { buildModeSelector, toggleBuildMode } from '../reducers/studio';
import { darkModeSelector } from '../reducers/settings';
import SettingsDialog from './SettingsDialog';

function Layout({ children }) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);
  const dispatch = useDispatch();
  const buildMode = useSelector(buildModeSelector);
  const darkMode = useSelector(darkModeSelector);

  useHotkeys('cmd+b', () => {
    dispatch(toggleBuildMode());
  });

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
                secondary: colors.red,
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

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <MuiAppBar position="fixed" elevation={2}>
        <Toolbar>
          <Grid justifyContent="space-between" alignItems="center" container>
            <Grid item>
              <Button
                startIcon={<DashboardIcon />}
                color="inherit"
                onClick={handleMenuOpen}
                variant="text"
              >
                Menu
              </Button>
              <Menu
                id="menu-appbar"
                anchorEl={anchorEl}
                anchorOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                keepMounted
                transformOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
              >
                <MenuItem onClick={handleMenuClose} selected>
                  <ListItemIcon>
                    <TableRowsIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText>Home</ListItemText>
                </MenuItem>
                <MenuItem onClick={handleMenuClose}>
                  <ListItemIcon>
                    <DonutSmallIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText>Dataset Analysis</ListItemText>
                </MenuItem>
                <Divider />
                <MenuItem
                  onClick={() => {
                    setSettingsOpen(true);
                    handleMenuClose();
                  }}
                >
                  <ListItemIcon>
                    <SettingsIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText>Settings</ListItemText>
                </MenuItem>
              </Menu>
            </Grid>
            <Grid item>
              <Grid container direction="row" alignItems="center">
                <BubbleChartIcon sx={{ marginRight: 1 }} />
                <Typography variant="h6" component="div">
                  REPSYS
                </Typography>
              </Grid>
            </Grid>
            <Grid item>
              <FormControlLabel
                control={
                  <Switch
                    color="secondary"
                    checked={buildMode}
                    sx={{
                      '& .MuiSwitch-track': {
                        backgroundColor: '#fff',
                      },
                    }}
                    onChange={() => dispatch(toggleBuildMode())}
                  />
                }
                label="Build Mode"
              />
            </Grid>
          </Grid>
        </Toolbar>
      </MuiAppBar>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          height: '100vh',
          overflow: 'auto',
          paddingTop: 3,
          paddingBottom: 4,
        }}
      >
        <Toolbar />
        {children}
      </Box>
      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </ThemeProvider>
  );
}

Layout.propTypes = {
  children: pt.node.isRequired,
};

export default Layout;
