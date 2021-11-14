import React from 'react';
import pt from 'prop-types';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import Box from '@mui/material/Box';
import RadioButtonCheckedIcon from '@mui/icons-material/RadioButtonChecked';
import Paper from '@mui/material/Paper';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import { useSelector, useDispatch } from 'react-redux';
import FavoriteIcon from '@mui/icons-material/Favorite';

import { getRequest } from './api';
import UserInteractions from './UserInteractions';
import {
  addUserToFavourites,
  removeUserFromFavourites,
  favouriteUsersSelector,
  sessionRecordingSelector,
  toggleSessionRecording,
  openSnackbar,
  setSelectedUser,
  customInteractionsSelector,
  selectedUserSelector,
  clearCustomInteractions,
} from '../reducers/studio';

function UserPanel({ onSearchClick }) {
  const dispatch = useDispatch();
  const favouriteUsers = useSelector(favouriteUsersSelector);
  const sessionRecord = useSelector(sessionRecordingSelector);
  const customInteractions = useSelector(customInteractionsSelector);
  const selectedUser = useSelector(selectedUserSelector);

  const { items: userData, isLoading: isUserLoading } = getRequest('/users');

  const handleSessionRecording = () => {
    dispatch(toggleSessionRecording());
    dispatch(setSelectedUser(null));
    if (!sessionRecord) {
      dispatch(
        openSnackbar({
          message: 'Recording started - click on items to interact.',
          severity: 'warning',
        })
      );
    }
  };

  const handleUserSelect = (event, user) => {
    dispatch(setSelectedUser(user));
    dispatch(clearCustomInteractions());
  };

  return (
    <Box sx={{ position: 'sticky', top: '4rem' }}>
      <Autocomplete
        disablePortal
        disabled={sessionRecord}
        value={selectedUser}
        loading={isUserLoading}
        onChange={handleUserSelect}
        isOptionEqualToValue={(option, value) => option.id === value.id}
        options={userData}
        getOptionLabel={(user) => `User ${user.label}`}
        sx={{ width: '100%', marginBottom: 2 }}
        renderInput={(params) => <TextField {...params} variant="filled" label="Selected user" />}
      />
      <Paper>
        <List>
          {selectedUser && (
            <ListItem disablePadding>
              {favouriteUsers.find((user) => user.id === selectedUser.id) !== undefined ? (
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
            <ListItemButton disabled={sessionRecord} onClick={onSearchClick}>
              <ListItemIcon>
                <PersonSearchIcon />
              </ListItemIcon>
              <ListItemText primary="Search options" />
            </ListItemButton>
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton onClick={handleSessionRecording}>
              <ListItemIcon>
                <RadioButtonCheckedIcon color={sessionRecord ? 'secondary' : 'inherit'} />
              </ListItemIcon>
              <ListItemText primary={sessionRecord ? 'Stop recording' : 'Record session'} />
            </ListItemButton>
          </ListItem>
        </List>
      </Paper>
      {(selectedUser || customInteractions.length > 0) && <UserInteractions />}
    </Box>
  );
}

UserPanel.propTypes = {
  onSearchClick: pt.func.isRequired,
};

export default UserPanel;
