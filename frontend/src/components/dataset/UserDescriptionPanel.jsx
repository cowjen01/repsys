import React, { useEffect, useState } from 'react';
import { Paper, Typography, Stack, Box, List } from '@mui/material';
import { useSelector } from 'react-redux';

import BarPlotHistogram from './BarPlotHistogram';
import PanelLoader from '../PanelLoader';
import { ItemListView } from '../items';
import { itemViewSelector } from '../../reducers/settings';
import { sleep } from '../../utils';

const data = {
  usersRatio: 0.45,
  interactions: {
    hist: [450, 230, 1000, 123],
    bins: [0, 5, 10, 50, 200],
  },
  items: [
    {
      title: 'Tom and Huck (1995)',
      about:
        'Two best friends witness a murder and embark on a series of adventures in order to prove the innocence of the man wrongly accused of the crime.',
      image:
        'https://m.media-amazon.com/images/M/MV5BN2ZkZTMxOTAtMzg1Mi00M2U0LWE2NWItZDg4YmQyZjVkMDdhXkEyXkFqcGdeQXVyNTM5NzI0NDY@..jpg',
      genres: 'Adventure, Comedy, Drama',
      languages: 'English',
      id: 6000,
    },
    {
      title: 'Dracula: Dead and Loving It (1995)',
      about: null,
      image:
        'https://m.media-amazon.com/images/M/MV5BZWQ0ZDFmYzMtZGMyMi00NmYxLWE0MGYtYzM2ZGNhMTE1NTczL2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMjM5ODMxODc@..jpg',
      genres: 'Comedy, Fantasy, Horror',
      languages: 'English, German',
      id: 3380,
    },
    {
      title: 'Cutthroat Island (1995)',
      about:
        'A female pirate and her companion race against their rivals to find a hidden island that contains a fabulous treasure.',
      image:
        'https://m.media-amazon.com/images/M/MV5BMDg2YTI0YmQtYzgwMi00Zjk4LWJkZjgtYjg0ZDE2ODUzY2RlL2ltYWdlXkEyXkFqcGdeQXVyNjQzNDI3NzY@..jpg',
      genres: 'Action, Adventure, Comedy',
      languages: 'English',
      id: 740,
    },
    {
      title: 'Sense and Sensibility (1995)',
      about:
        'Rich Mr. Dashwood dies, leaving his second wife and her three daughters poor by the rules of inheritance. The two eldest daughters are the title opposites.',
      image:
        'https://m.media-amazon.com/images/M/MV5BNzk1MjU3MDQyMl5BMl5BanBnXkFtZTcwNjc1OTM2MQ@@..jpg',
      genres: 'Drama, Romance',
      languages: 'English, French',
      id: 391,
    },
    {
      title: 'Now and Then (1995)',
      about:
        'Four 12-year-old girls grow up together during an eventful small-town summer in 1970.',
      image:
        'https://m.media-amazon.com/images/M/MV5BMTM2MDQ1YjUtMGM0NC00NmFlLTljMDktZjJiNWRhMWYxOWYyXkEyXkFqcGdeQXVyNjgzMjI4ODE@..jpg',
      genres: 'Comedy, Drama, Romance',
      languages: 'English',
      id: 2874,
    },
  ],
};

function UserDescriptionPanel({ userIds }) {
  const [isLoading, setIsLoading] = useState(false);
  const itemView = useSelector(itemViewSelector);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      await sleep(500);
      setIsLoading(false);
    }

    if (userIds.length) {
      loadData();
    }
  }, [userIds]);

  if (!userIds.length) {
    return null;
  }

  if (isLoading) {
    return <PanelLoader />;
  }

  return (
    <Paper sx={{ p: 2, maxHeight: '100%', overflow: 'auto' }}>
      <Stack spacing={2}>
        <Box>
          <Typography variant="h6" sx={{ fontSize: '1.1rem' }}>
            Interacted Items
          </Typography>
          <Typography gutterBottom variant="body2">
            A list of the most interacted items
          </Typography>
          <List dense>
            {data.items.map((item) => (
              <ItemListView
                key={item.id}
                id={item.id}
                title={item[itemView.title]}
                subtitle={item[itemView.subtitle]}
                image={item[itemView.image]}
                style={{ paddingLeft: 5 }}
              />
            ))}
          </List>
        </Box>
        <Box>
          <Typography variant="h6" sx={{ fontSize: '1.1rem' }}>
            Interactions Distribution
          </Typography>
          <Typography gutterBottom variant="body2">
            A distribution of total interactions made by users
          </Typography>
          <BarPlotHistogram bins={data.interactions.bins} hist={data.interactions.hist} />
        </Box>
      </Stack>
    </Paper>
  );
}

export default UserDescriptionPanel;
