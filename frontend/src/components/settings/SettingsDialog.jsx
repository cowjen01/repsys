import React from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Switch,
  FormControlLabel,
  FormGroup,
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';

import { darkModeSelector, toggleDarkMode } from '../../reducers/settings';
import { closeSettingsDialog, settingsDialogSelector } from '../../reducers/dialogs';

function SettingsDialog() {
  const darkMode = useSelector(darkModeSelector);
  const dialogOpen = useSelector(settingsDialogSelector);
  const dispatch = useDispatch();

  const handleClose = () => {
    dispatch(closeSettingsDialog());
  };

  return (
    <Dialog open={dialogOpen} fullWidth maxWidth="sm" onClose={handleClose}>
      <DialogTitle>Repsys Settings</DialogTitle>
      <DialogContent>
        <FormGroup>
          <FormControlLabel
            control={
              <Switch
                color="primary"
                checked={darkMode}
                onChange={() => dispatch(toggleDarkMode())}
              />
            }
            label="Dark Mode"
          />
        </FormGroup>
      </DialogContent>
      <DialogActions>
        <Button color="secondary" onClick={handleClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default SettingsDialog;
