import React, { Component } from 'react';
import axios from 'axios';


/**
 * CNN: https://www.cnn.com/2021/02/05/media/lou-dobbs-fox-show-canceled/index.html
 * FOX: https://www.foxnews.com/politics/biden-terrorist-designation-yemens-houthi-militia
 * HUFFPOST: https://www.huffpost.com/entry/covid-19-eviction-crisis-women_n_5fca8af3c5b626e08a29de11
 */

export default class ScrapeInput extends Component {
    constructor(props){
        super(props);
        this.state = {
            input_url: '',
            output: {},
        }
        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }
    handleChange(event){
        this.setState({input_url: event.target.value})
    }
    handleSubmit(event){
        event.preventDefault();
        this.props.handleSubmit(this.state.input_url)
    }
    render(){
        return(
            <div className='url-input-wrapper'>
                <input type='text' onChange={this.handleChange} value={this.state.input_url} placeholder='enter url'></input>
                <button type='submit' onClick={this.handleSubmit}>Submit</button>
            </div>
        );
    }
}