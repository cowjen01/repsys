import { useState, useEffect } from 'react';

export function fetchPredictions(body) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState();
  const [isLoading, setIsLoading] = useState(true);

  const encodedBody = JSON.stringify(body);

  useEffect(() => {
    let isActive = true;

    setIsLoading(true);

    fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: encodedBody,
    })
      .then((response) => {
        if (!response.ok) {
          return Promise.reject(response);
        }
        return response.json();
      })
      .then((data) => {
        if (isActive) {
          setItems(data);
          setIsLoading(false);
        }
      })
      .catch((response) => {
        response.json().then((data) => {
          setIsLoading(false);
          setError(data.message);
        })
      });

    return () => {
      isActive = false;
    };
  }, [encodedBody]);

  return { items, isLoading, error };
}
