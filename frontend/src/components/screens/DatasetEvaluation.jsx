import React, { useState, useMemo, useEffect, useRef } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Stack,
  List,
  Box,
  Tabs,
  Tab,
  ListItem,
  Alert,
  ListSubheader,
  ListItemText,
  Chip,
} from '@mui/material';
import Plotly from 'plotly.js';

import { IndicatorPlot, HistogramPlot, ScatterPlot, BarPlot } from '../plots';
import { ItemListView } from '../items';
import TabPanel from '../TabPanel';
import { CategoryFilter } from '../filters';

// select fields of type: Title, Category, Tags, Number
const itemEmbeddings = [
  {
    x: 0.72,
    y: 0.7,
    c: 3,
    label: 'Ghosts of Mississippi',
    year: 1968,
    country: 'CO',
    genres: ['comedy', 'action'],
  },
  {
    x: 0.37,
    y: 0.78,
    c: 1,
    label: 'American Pie 2',
    year: 1996,
    country: 'CO',
    genres: ['action'],
  },
  {
    x: 0.07,
    y: 0.44,
    c: 5,
    label: 'Inglorious Bastards (Quel maledetto treno blindato)',
    year: 1997,
    country: 'MK',
    genres: ['comedy', 'horror', 'action'],
  },
  {
    x: 0.07,
    y: 0.93,
    c: 3,
    label: 'Prince of Egypt, The',
    year: 2001,
    country: 'FR',
    genres: ['action'],
  },
  {
    x: 0.63,
    y: 0.79,
    c: 3,
    label: 'Friday the 13th',
    year: 2010,
    country: 'CN',
    genres: ['horror'],
  },
  {
    x: 0.65,
    y: 0.4,
    c: 2,
    label: 'Pearl Harbor',
    year: 2015,
    country: 'ID',
    genres: ['drama'],
  },
  {
    x: 0.42,
    y: 0.79,
    c: 3,
    label: 'Ghosts of Mississippi',
    year: 1968,
    country: 'CO',
    genres: ['comedy', 'action'],
  },
  {
    x: 0.17,
    y: 0.98,
    c: 1,
    label: 'American Pie 2',
    year: 1996,
    country: 'CO',
    genres: ['action'],
  },
  {
    x: 0.37,
    y: 0.84,
    c: 5,
    label: 'Inglorious Bastards (Quel maledetto treno blindato)',
    year: 1997,
    country: 'MK',
    genres: ['comedy', 'horror', 'action'],
  },
  {
    x: 0.27,
    y: 0.43,
    c: 3,
    label: 'Prince of Egypt, The',
    year: 2001,
    country: 'FR',
    genres: ['action'],
  },
  {
    x: 0.43,
    y: 0.29,
    c: 3,
    label: 'Friday the 13th',
    year: 2010,
    country: 'CN',
    genres: ['horror'],
  },
  {
    x: 0.6,
    y: 0.44,
    c: 2,
    label: 'Pearl Harbor',
    year: 2015,
    country: 'ID',
    genres: ['drama'],
  },
];

const columns = {
  year: {
    dtype: 'number',
    bins: [
      [0, 1983],
      [1984, 2000],
      [2001, 2005],
      [2006, 2020],
    ],
  },
  genres: {
    dtype: 'tags',
    options: ['action', 'drama', 'comedy', 'horror'],
  },
  country: {
    dtype: 'category',
    options: ['CO', 'MK', 'CN', 'FR', 'ID'],
  },
};

const characteristics = {
  genres: {
    topValues: ['action', 'drama'],
  },
  country: {
    topValues: ['CO', 'MK', 'FR'],
  },
  year: {
    hist: [2, 4, 2, 10, 12],
    bins: [0, 2000, 2006, 2020, 2050, 2070],
  },
};

const activeColor = '#5c6bc0';
const defaultColor = '#e8eaf6';

function DatasetEvaluation() {
  const [selectedField, setSelectedField] = useState('');
  const [selectedValues, setSelectedValues] = useState([]);
  const [highlightedItems, setHighlightedItems] = useState([]);
  const [selectedPoints, setSelectedPoints] = useState([]);

  const scatterRef = useRef();

  useEffect(() => {
    if (selectedField && selectedValues.length) {
      const indices = itemEmbeddings.reduce((acc, item, index) => {
        const fieldType = columns[selectedField].dtype;
        const value = item[selectedField];
        if (value) {
          if (fieldType === 'number') {
            const bin = columns[selectedField].bins[selectedValues[0]];
            if (value >= bin[0] && value <= bin[1]) {
              acc.push(index);
            }
          } else if (fieldType === 'category') {
            if (selectedValues.includes(value)) {
              acc.push(index);
            }
          } else if (fieldType === 'tags') {
            const intersection = value.filter((x) => selectedValues.includes(x));
            if (intersection.length > 0) {
              acc.push(index);
            }
          }
        }
        return acc;
      }, []);
      setHighlightedItems(indices);
    }
  }, [selectedField, selectedValues]);

  const scatterPoints = useMemo(
    () => ({
      x: itemEmbeddings.map(({ x }) => x),
      y: itemEmbeddings.map(({ y }) => y),
      label: itemEmbeddings.map(({ label }) => label),
    }),
    []
  );

  const scatterColors = useMemo(() => {
    if (!highlightedItems.length) {
      return activeColor;
    }

    const colors = [];
    for (let i = 0; i < itemEmbeddings.length; i += 1) {
      colors.push(defaultColor);
    }

    highlightedItems.forEach((p) => {
      colors[p] = activeColor;
    });

    return colors;
  }, [highlightedItems]);

  const filterOptions = useMemo(() => {
    if (selectedField) {
      const col = columns[selectedField];
      if (col.dtype === 'number') {
        return col.bins.map((bin, index) => ({
          value: index,
          label: `${bin[0]} - ${bin[1]}`,
        }));
      }
      return col.options.map((option) => ({
        value: option,
        label: option,
      }));
    }
    return [];
  }, [selectedField]);

  const isMultipleSelect = useMemo(() => {
    if (selectedField) {
      const fieldType = columns[selectedField].dtype;
      return fieldType === 'tags' || fieldType === 'category';
    }
    return false;
  }, [selectedField]);

  const resetScatterSelection = () => {
    setSelectedPoints([]);
    Plotly.restyle(scatterRef.current.el, { selectedpoints: [null] });
  };

  return (
    <Container maxWidth="xl">
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Box pl={1}>
            <Typography component="div" variant="h6">
              Item Embeddings
            </Typography>
            <Typography variant="subtitle1" gutterBottom>
              A space of item embeddings and their attributes to explore the primary space
            </Typography>
          </Box>
          <Grid container spacing={2}>
            <Grid item md={7} xs={12}>
              <Paper sx={{ p: 2 }}>
                <Stack direction="row" spacing={2}>
                  <CategoryFilter
                    label="Item attribute"
                    displayEmpty
                    value={selectedField}
                    onChange={(newValue) => {
                      setSelectedField(newValue);
                      setSelectedValues([]);
                      setHighlightedItems([]);
                      resetScatterSelection();
                    }}
                    options={Object.keys(columns).map((col) => ({
                      value: col,
                      label: col,
                    }))}
                  />
                  <CategoryFilter
                    label="Attribute value"
                    disabled={!selectedField}
                    value={isMultipleSelect ? selectedValues : selectedValues[0]}
                    multiple={isMultipleSelect}
                    onChange={(newValue) => {
                      if (isMultipleSelect) {
                        setSelectedValues(newValue);
                      } else {
                        setSelectedValues([newValue]);
                      }
                    }}
                    options={filterOptions}
                  />
                </Stack>
                <ScatterPlot
                  height={450}
                  x={scatterPoints.x}
                  y={scatterPoints.y}
                  color={scatterColors}
                  label={scatterPoints.label}
                  innerRef={scatterRef}
                  onDeselect={() => {
                    setSelectedValues([]);
                    setSelectedField('');
                    setHighlightedItems([]);
                    resetScatterSelection();
                  }}
                  onSelected={(eventData) => {
                    if (eventData && eventData.points.length) {
                      const { points } = eventData;
                      const selected = points[0].data.selectedpoints;
                      setSelectedPoints(selected);
                      setHighlightedItems(selected);
                      setSelectedValues([]);
                      setSelectedField('');
                    }
                  }}
                />
              </Paper>
            </Grid>
            <Grid item xs={12} md={5}>
              <Paper sx={{ p: 2 }}>
                {selectedPoints.length > 0 ? (
                  <Stack spacing={2}>
                    <Typography>Selected {selectedPoints.length} items</Typography>
                    {Object.entries(characteristics).map(([col, data]) => (
                      <Box key={col}>
                        <Typography>{col}</Typography>
                        {(columns[col].dtype === 'tags' || columns[col].dtype === 'category') && (
                          <>
                            <Typography>Top values</Typography>
                            <Stack direction="row" spacing={1}>
                              {data.topValues.map((value) => (
                                <Chip key={value} label={value} />
                              ))}
                            </Stack>
                          </>
                        )}
                        {columns[col].dtype === 'number' && (
                          <>
                            <Typography>Values distribution</Typography>
                            <BarPlot
                              height={150}
                              data={[
                                {
                                  x: data.hist.map(
                                    (_, index) => `${data.bins[index]} - ${data.bins[index + 1]}`
                                  ),
                                  y: data.hist,
                                },
                              ]}
                            />
                          </>
                        )}
                      </Box>
                    ))}
                  </Stack>
                ) : (
                  <Alert severity="info">
                    To see the details, select a range of items in the scatter plot.
                  </Alert>
                )}
              </Paper>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Container>
  );
}

export default DatasetEvaluation;
