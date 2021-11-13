import React from 'react';
import pt from 'prop-types';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import { Typography, Box, Button, Grid } from '@mui/material';
import Checkbox from '@mui/material/Checkbox';
import FavoriteBorder from '@mui/icons-material/FavoriteBorder';
import Favorite from '@mui/icons-material/Favorite';
import RadioButtonCheckedIcon from '@mui/icons-material/RadioButtonChecked';
import Paper from '@mui/material/Paper';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import ListSubheader from '@mui/material/ListSubheader';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import { useSelector, useDispatch } from 'react-redux';
import FavoriteIcon from '@mui/icons-material/Favorite';

import { getRequest } from './api';
import UserInteractions from './UserInteractions';
import {
  addUserToFavourites,
  removeUserFromFavourites,
  favouriteUsersSelector,
} from '../reducers/studio';

function UserPanel({
  selectedUser,
  onUserSelect,
  onSearchClick,
  customInteractions,
  onInteractionsDelete,
}) {
  const { items: userData, isLoading: isUserLoading } = getRequest('/users');
  const dispatch = useDispatch();
  const favouriteUsers = useSelector(favouriteUsersSelector);

  return (
    <Box sx={{ position: 'sticky', top: '4rem' }}>
      {/* <Typography variant="h6" component="div" gutterBottom>
        User Selector
      </Typography> */}
      {/* <Grid container alignItems="center" sx={{ marginBottom: 1 }}>
        <Grid item>
          <Typography variant="h6" component="div">
            {selectedUser ? `User ${selectedUser.label}` : `User Selector`}
          </Typography>
        </Grid>
        {selectedUser && (
          <Grid item>
            <Checkbox color="secondary" icon={<FavoriteBorder />} checkedIcon={<Favorite />} />
          </Grid>
        )}
      </Grid> */}
      <Autocomplete
        disablePortal
        value={selectedUser}
        onChange={(event, newValue) => {
          onUserSelect(newValue);
        }}
        isOptionEqualToValue={(option, value) => option.id === value.id}
        options={userData}
        getOptionLabel={(user) => `User ${user.label}`}
        sx={{ width: '100%', marginBottom: 2 }}
        renderInput={(params) => <TextField {...params} variant="filled" label="Selected user" />}
      />
      {/* <Paper>
<Button
        onClick={onSearchClick}
        startIcon={<PersonSearchIcon />}
        // variant="contained"
        color="secondary"
      >
        More Options
      </Button>
      <Button
        onClick={onSearchClick}
        startIcon={<RadioButtonCheckedIcon />}
        // variant="contained"
        color="secondary"
      >
        Record Session
      </Button>
      </Paper> */}
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
            <ListItemButton onClick={onSearchClick}>
              <ListItemIcon>
                <PersonSearchIcon />
              </ListItemIcon>
              <ListItemText primary="Open search options" />
            </ListItemButton>
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton>
              <ListItemIcon>
                <RadioButtonCheckedIcon />
              </ListItemIcon>
              <ListItemText primary="Record new session" />
            </ListItemButton>
          </ListItem>
        </List>
      </Paper>

      {(selectedUser || customInteractions.length > 0) && (
        <UserInteractions
          selectedUser={selectedUser}
          onInteractionsDelete={onInteractionsDelete}
          customInteractions={customInteractions}
        />
      )}
    </Box>
  );
}

UserPanel.defaultProps = {
  selectedUser: null,
};

UserPanel.propTypes = {
  selectedUser: pt.shape({
    id: pt.number,
    label: pt.string,
  }),
  onUserSelect: pt.func.isRequired,
  onSearchClick: pt.func.isRequired,
};

export default UserPanel;
