import React from 'react';
import {
  Box,
  Chip,
  Switch,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Paper,
} from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import FilterListIcon from '@mui/icons-material/FilterList';
import RadioButtonCheckedIcon from '@mui/icons-material/RadioButtonChecked';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import BuildIcon from '@mui/icons-material/Build';

import {
  sessionRecordingSelector,
  customInteractionsSelector,
  selectedUserSelector,
  setCustomInteractions,
  toggleSessionRecording,
  buildModeSelector,
  toggleBuildMode,
  setSelectedUser
} from '../../reducers/root';
import InteractionsList from './InteractionsList';
import { openSnackbar, openUserSelectDialog } from '../../reducers/dialogs';
import { interactionsSelector } from '../../reducers/interactions';

function UserPanel() {
  const dispatch = useDispatch();
  const sessionRecord = useSelector(sessionRecordingSelector);
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const buildMode = useSelector(buildModeSelector);
  const userInteractions = useSelector(interactionsSelector);

  const handleDelete = () => {
    dispatch(setCustomInteractions([]));
    dispatch(setSelectedUser(null))
  };

  const handleRecordBtnClick = () => {
    if (selectedUser) {
      dispatch(toggleSessionRecording(userInteractions));
    } else {
      dispatch(toggleSessionRecording());
    }
    if (!sessionRecord) {
      dispatch(
        openSnackbar({
          message: 'Recording started - click on items to interact.',
          severity: 'warning',
        })
      );
    }
  };

  const handleSelectBtnClick = () => {
    dispatch(openUserSelectDialog());
  };

  return (
    <Box sx={{ position: 'sticky', top: '4rem' }}>
      <Paper>
        <List>
          <ListItem disabled={sessionRecord}>
            <ListItemIcon>
              <BuildIcon />
            </ListItemIcon>
            <ListItemText primary="Build mode" />
            <Switch edge="end" onChange={() => dispatch(toggleBuildMode())} checked={buildMode} />
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton disabled={sessionRecord} onClick={handleSelectBtnClick}>
              <ListItemIcon>
                <PersonSearchIcon />
              </ListItemIcon>
              <ListItemText primary="User selection" />
            </ListItemButton>
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton onClick={handleRecordBtnClick}>
              <ListItemIcon>
                <RadioButtonCheckedIcon color={sessionRecord ? 'secondary' : 'inherit'} />
              </ListItemIcon>
              <ListItemText primary={sessionRecord ? 'Stop recording' : 'Session record'} />
            </ListItemButton>
          </ListItem>
        </List>
      </Paper>
      {(selectedUser || customInteractions.length > 0) && (
        <Box sx={{ marginTop: 2 }}>
          {customInteractions.length > 0 ? (
            <Chip
              sx={{ marginBottom: 2 }}
              onDelete={handleDelete}
              icon={<FilterListIcon />}
              label="Custom interactions"
            />
          ) : (
            <Chip
              sx={{ marginBottom: 2 }}
              onDelete={handleDelete}
              icon={<FilterListIcon />}
              label={`User ${selectedUser}`}
            />
          )}
          <InteractionsList />
        </Box>
      )}
    </Box>
  );
}

export default UserPanel;
