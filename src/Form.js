import axios from 'axios';
import React from 'react';
import './Form.css';

class Form extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            text: "Enter text here",
            response: []
        };

        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }
    handleChange(event){
        this.setState({text: event.target.value}); //, response: event.target.response});
    }
    handleSubmit(event){
        event.preventDefault();
        const myText = this.state.text;
        console.log("Input: " + myText);
        axios.post("/result", {myText})
            .then(res => {
                this.setState({ response: res.data.sentence_results })
                //this.setState({response: res.data});//.data.text});
                console.log(res.data.sentence_results[0][0]);//.data.text);
            });
    }
    render(){
        console.log(this.state.response[0])
        var wordList = this.state.response.map((item) =>
            <section className="results_display">
                <ul id="results">
                        <li className="word">
                            {item.words.map((word) =>
                                <ul className="list-group list-group-flush">
                                    <li className="list-group-item" key={word}>
                                        {word}
                                    </li>
                                </ul>
                            )}
                        </li>
                        <li className="other_info" key={item.average}>
                            <p>
                                Average Score: <strong>{item.average}</strong>.
                            </p>
                            <p>
                                The most biased word is: <strong>{item.max_biased_word}</strong>.
                            </p>
                        </li>
                </ul>
            </section>
            )
        return(
            <div>
                <form className="form" onSubmit={this.handleSubmit}>
                <p></p>
                    <label>
                        <strong>Text:</strong> <input className="text_input" key="text-input" type="text" value={this.state.text} onChange={this.handleChange} />
                    </label>
                    <button className="submit_button" type="submit" value="Submit">Submit</button>
                    <ul className="results_wrapper">{wordList}</ul>
                </form>
                <p></p>
            </div>
        );
    }
}
export default Form;