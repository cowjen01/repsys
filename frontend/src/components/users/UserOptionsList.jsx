import React from 'react';
import { Paper, List, ListItem, ListItemButton, ListItemIcon, ListItemText } from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import FavoriteIcon from '@mui/icons-material/Favorite';
import RadioButtonCheckedIcon from '@mui/icons-material/RadioButtonChecked';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';

import {
  addUserToFavourites,
  removeUserFromFavourites,
  favouriteUsersSelector,
  sessionRecordingSelector,
  toggleSessionRecording,
  setSelectedUser,
  selectedUserSelector,
} from '../../reducers/root';
import { openSnackbar, openUserSelectDialog } from '../../reducers/dialogs';

function UserOptionsList() {
  const dispatch = useDispatch();
  const favouriteUsers = useSelector(favouriteUsersSelector);
  const sessionRecord = useSelector(sessionRecordingSelector);
  const selectedUser = useSelector(selectedUserSelector);

  const handleRecordBtnClick = () => {
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

  const handleSelectBtnClick = () => {
    dispatch(openUserSelectDialog());
  };

  return (
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
      </List>
    </Paper>
  );
}

export default UserOptionsList;
