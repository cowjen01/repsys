import React from 'react';
import { Box, Toolbar, AppBar, Typography, Stack, Button, Grid, Container } from '@mui/material';
import { useDispatch } from 'react-redux';
import BubbleChartIcon from '@mui/icons-material/BubbleChart';
import DonutSmallIcon from '@mui/icons-material/DonutSmall';
import TableRowsIcon from '@mui/icons-material/TableRows';
import SettingsIcon from '@mui/icons-material/Settings';
import { Link as RouterLink, Outlet } from 'react-router-dom';
import EqualizerIcon from '@mui/icons-material/Equalizer';

import { openSettingsDialog } from '../reducers/dialogs';

function Layout() {
  const dispatch = useDispatch();

  return (
    <>
      <AppBar position="fixed" elevation={2} color="secondary">
        <Toolbar>
          <Grid container direction="row" alignItems="center">
            <BubbleChartIcon sx={{ marginRight: 1 }} />
            <Typography variant="h6" component="div">
              REPSYS
            </Typography>
          </Grid>
          <Stack spacing={2} direction="row">
            <Button
              to="/"
              component={RouterLink}
              startIcon={<TableRowsIcon />}
              color="inherit"
              variant="text"
            >
              Previews
            </Button>
            <Button
              to="/models"
              component={RouterLink}
              startIcon={<EqualizerIcon />}
              color="inherit"
              variant="text"
            >
              Evaluations
            </Button>
            <Button
              to="/dataset"
              component={RouterLink}
              startIcon={<DonutSmallIcon />}
              color="inherit"
              variant="text"
            >
              Dataset
            </Button>
            <Button
              startIcon={<SettingsIcon />}
              color="inherit"
              variant="text"
              onClick={() => {
                dispatch(openSettingsDialog());
              }}
            >
              Settings
            </Button>
          </Stack>
        </Toolbar>
      </AppBar>
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
        <Container maxWidth="xl">
          <Outlet />
        </Container>
      </Box>
    </>
  );
}

export default Layout;
