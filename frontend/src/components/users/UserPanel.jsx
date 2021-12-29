import React from 'react';
import {
  TextField,
  Autocomplete,
  Box,
  Chip,
  FormControlLabel,
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
import FavoriteIcon from '@mui/icons-material/Favorite';
import RadioButtonCheckedIcon from '@mui/icons-material/RadioButtonChecked';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import WifiIcon from '@mui/icons-material/Wifi';
import BuildIcon from '@mui/icons-material/Build';

import {
  sessionRecordingSelector,
  setSelectedUser,
  customInteractionsSelector,
  selectedUserSelector,
  setCustomInteractions,
  toggleSessionRecording,
  buildModeSelector,
  favouriteUsersSelector,
  toggleBuildMode,
  removeUserFromFavourites,
  addUserToFavourites,
} from '../../reducers/root';
import { usersSelector, usersStatusSelector } from '../../reducers/users';
import InteractionsList from './InteractionsList';
import { openSnackbar, openUserSelectDialog } from '../../reducers/dialogs';
import { interactionsSelector } from '../../reducers/interactions';

function UserPanel() {
  const dispatch = useDispatch();
  const sessionRecord = useSelector(sessionRecordingSelector);
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);
  const usersData = useSelector(usersSelector);
  const usersStatus = useSelector(usersStatusSelector);
  const buildMode = useSelector(buildModeSelector);
  const favouriteUsers = useSelector(favouriteUsersSelector);
  const userInteractions = useSelector(interactionsSelector);

  const handleUserSelect = (event, user) => {
    dispatch(setSelectedUser(user));
    dispatch(setCustomInteractions([]));
  };

  const handleDelete = () => {
    dispatch(setCustomInteractions([]));
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
      <Autocomplete
        disablePortal
        disabled={sessionRecord}
        value={selectedUser}
        loading={usersStatus === 'loading'}
        onChange={handleUserSelect}
        options={usersData}
        getOptionLabel={(user) => `User ${user}`}
        sx={{ width: '100%', marginBottom: 2 }}
        renderInput={(params) => <TextField {...params} variant="filled" label="Selected user" />}
      />
      <Paper>
        {/* <FormControlLabel
        sx={{ marginBottom: 1 }}
        control={
          <Switch
            color="secondary"
            checked={buildMode}
            onChange={() => dispatch(toggleBuildMode())}
          />
        }
        label={buildMode ? 'Build Mode' : 'Preview Mode'}
      /> */}
        <List dense>
          {selectedUser && (
            <ListItem disablePadding>
              {favouriteUsers.includes(selectedUser) ? (
                <ListItemButton onClick={() => dispatch(removeUserFromFavourites())}>
                  <ListItemIcon>
                    <FavoriteIcon color="secondary" />
                  </ListItemIcon>
                  <ListItemText primary="Remove from favourites" />
                </ListItemButton>
              ) : (
                <ListItemButton onClick={() => dispatch(addUserToFavourites())}>
                  <ListItemIcon>
                    <FavoriteBorderIcon />
                  </ListItemIcon>
                  <ListItemText primary="Add to favourites" />
                </ListItemButton>
              )}
            </ListItem>
          )}
          <ListItem disablePadding>
            <ListItemButton disabled={sessionRecord} onClick={handleSelectBtnClick}>
              <ListItemIcon>
                <PersonSearchIcon />
              </ListItemIcon>
              <ListItemText primary="Search options" />
            </ListItemButton>
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton onClick={handleRecordBtnClick}>
              <ListItemIcon>
                <RadioButtonCheckedIcon color={sessionRecord ? 'secondary' : 'inherit'} />
              </ListItemIcon>
              <ListItemText primary={sessionRecord ? 'Stop recording' : 'Record session'} />
            </ListItemButton>
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <BuildIcon />
            </ListItemIcon>
            <ListItemText primary="Build mode" />
            <Switch edge="end" onChange={() => dispatch(toggleBuildMode())} checked={buildMode} />
          </ListItem>
        </List>
      </Paper>
      {(selectedUser || customInteractions.length > 0) && (
        <Box sx={{ marginTop: 2 }}>
          {customInteractions.length > 0 && (
            <Chip
              sx={{ marginBottom: 2 }}
              onDelete={handleDelete}
              icon={<FilterListIcon />}
              label="Custom interactions"
            />
          )}
          <InteractionsList />
        </Box>
      )}
    </Box>
  );
}

export default UserPanel;
